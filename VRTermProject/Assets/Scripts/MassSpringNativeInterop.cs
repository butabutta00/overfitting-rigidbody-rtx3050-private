using System;
using System.Runtime.InteropServices;
using UnityEngine;

internal static class MassSpringNativeInterop
{
    private const string LibraryName = "mass_spring_native";

    [StructLayout(LayoutKind.Sequential)]
    internal struct SemiParams
    {
        public float dt;
        public float springStiffness;
        public float springDamping;
        public float gravityX;
        public float gravityY;
        public float gravityZ;
        public float velocityDamping;
        public int substeps;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct ImplicitParams
    {
        public float dt;
        public float springStiffness;
        public float springDamping;
        public float gravityX;
        public float gravityY;
        public float gravityZ;
        public float velocityDamping;
        public int implicitIterations;
        public float cgTolerance;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct OneDState
    {
        public float position;
        public float velocity;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct OneDParams
    {
        public float dt;
        public float mass;
        public float stiffness;
        public float damping;
    }

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr mssCreateSystem();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssDestroySystem(IntPtr system);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssUploadTopology(
        IntPtr system,
        int particleCount,
        float[] positionXYZ,
        float[] masses,
        byte[] fixedMask,
        int springCount,
        int[] springEndpoints,
        float[] restLengths);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssStepSemi(IntPtr system, ref SemiParams stepParams);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssStepImplicit(IntPtr system, ref ImplicitParams stepParams);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssDownloadState(IntPtr system, float[] outPositionXYZ, float[] outVelocityXYZ, int particleCount);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int mssOneDImplicitStep(ref OneDState state, ref OneDParams simulationParams);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr mssGetLastError();

    internal static string GetLastErrorMessage()
    {
        try
        {
            IntPtr ptr = mssGetLastError();
            if (ptr == IntPtr.Zero)
            {
                return "unknown native error";
            }

            return Marshal.PtrToStringAnsi(ptr) ?? "unknown native error";
        }
        catch (Exception ex)
        {
            return ex.Message;
        }
    }

    internal sealed class SystemHandle : IDisposable
    {
        private IntPtr nativePtr;
        private readonly float[] positionBuffer;
        private readonly float[] velocityBuffer;
        private bool disposed;

        private SystemHandle(IntPtr ptr, int particleCount)
        {
            nativePtr = ptr;
            positionBuffer = new float[particleCount * 3];
            velocityBuffer = new float[particleCount * 3];
        }

        internal static bool TryCreate(
            Vector3[] initialPositions,
            float[] masses,
            byte[] fixedMask,
            int[] springEndpoints,
            float[] restLengths,
            out SystemHandle handle,
            out string error)
        {
            handle = null;
            error = string.Empty;

            try
            {
                IntPtr ptr = mssCreateSystem();
                if (ptr == IntPtr.Zero)
                {
                    error = GetLastErrorMessage();
                    return false;
                }

                int particleCount = initialPositions.Length;
                int springCount = restLengths.Length;
                float[] packedPos = new float[particleCount * 3];
                for (int i = 0; i < particleCount; i++)
                {
                    int baseIndex = i * 3;
                    packedPos[baseIndex] = initialPositions[i].x;
                    packedPos[baseIndex + 1] = initialPositions[i].y;
                    packedPos[baseIndex + 2] = initialPositions[i].z;
                }

                int result = mssUploadTopology(ptr, particleCount, packedPos, masses, fixedMask, springCount, springEndpoints, restLengths);
                if (result != 0)
                {
                    error = GetLastErrorMessage();
                    mssDestroySystem(ptr);
                    return false;
                }

                handle = new SystemHandle(ptr, particleCount);
                return true;
            }
            catch (DllNotFoundException ex)
            {
                error = ex.Message;
                return false;
            }
            catch (EntryPointNotFoundException ex)
            {
                error = ex.Message;
                return false;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        internal bool StepSemi(float dt, float springStiffness, float springDamping, Vector3 gravity, float velocityDamping, int substeps)
        {
            if (disposed || nativePtr == IntPtr.Zero)
            {
                return false;
            }

            SemiParams stepParams = new SemiParams
            {
                dt = dt,
                springStiffness = springStiffness,
                springDamping = springDamping,
                gravityX = gravity.x,
                gravityY = gravity.y,
                gravityZ = gravity.z,
                velocityDamping = velocityDamping,
                substeps = Mathf.Max(1, substeps)
            };

            return mssStepSemi(nativePtr, ref stepParams) == 0;
        }

        internal bool StepImplicit(float dt, float springStiffness, float springDamping, Vector3 gravity, float velocityDamping, int implicitIterations, float cgTolerance)
        {
            if (disposed || nativePtr == IntPtr.Zero)
            {
                return false;
            }

            ImplicitParams stepParams = new ImplicitParams
            {
                dt = dt,
                springStiffness = springStiffness,
                springDamping = springDamping,
                gravityX = gravity.x,
                gravityY = gravity.y,
                gravityZ = gravity.z,
                velocityDamping = velocityDamping,
                implicitIterations = Mathf.Max(1, implicitIterations),
                cgTolerance = Mathf.Max(1e-12f, cgTolerance)
            };

            return mssStepImplicit(nativePtr, ref stepParams) == 0;
        }

        internal bool DownloadState(Vector3[] positions, Vector3[] velocities)
        {
            if (disposed || nativePtr == IntPtr.Zero)
            {
                return false;
            }

            if (mssDownloadState(nativePtr, positionBuffer, velocityBuffer, positions.Length) != 0)
            {
                return false;
            }

            for (int i = 0; i < positions.Length; i++)
            {
                int baseIndex = i * 3;
                positions[i] = new Vector3(positionBuffer[baseIndex], positionBuffer[baseIndex + 1], positionBuffer[baseIndex + 2]);
                velocities[i] = new Vector3(velocityBuffer[baseIndex], velocityBuffer[baseIndex + 1], velocityBuffer[baseIndex + 2]);
            }

            return true;
        }

        internal bool DownloadPositions(Vector3[] positions)
        {
            if (disposed || nativePtr == IntPtr.Zero)
            {
                return false;
            }

            if (mssDownloadState(nativePtr, positionBuffer, null, positions.Length) != 0)
            {
                return false;
            }

            for (int i = 0; i < positions.Length; i++)
            {
                int baseIndex = i * 3;
                positions[i] = new Vector3(positionBuffer[baseIndex], positionBuffer[baseIndex + 1], positionBuffer[baseIndex + 2]);
            }

            return true;
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            if (nativePtr != IntPtr.Zero)
            {
                mssDestroySystem(nativePtr);
                nativePtr = IntPtr.Zero;
            }
        }
    }

    internal static bool TryStepOneDImplicit(ref float positionX, ref float velocityX, float dt, float mass, float stiffness, float damping)
    {
        try
        {
            OneDState state = new OneDState { position = positionX, velocity = velocityX };
            OneDParams simulationParams = new OneDParams
            {
                dt = dt,
                mass = mass,
                stiffness = stiffness,
                damping = damping
            };

            int result = mssOneDImplicitStep(ref state, ref simulationParams);
            if (result != 0)
            {
                return false;
            }

            positionX = state.position;
            velocityX = state.velocity;
            return true;
        }
        catch
        {
            return false;
        }
    }
}
