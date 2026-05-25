using System.Diagnostics;
using System.Globalization;
using System.Numerics;

const float Epsilon = 1e-6f;

var options = CliOptions.Parse(args);

var system = CpuMassSpringSystem.CreateChain(
    options.ParticleCount,
    options.Spacing,
    options.Mass,
    options.FixedFirstParticle);

system.SetInitialVelocity(options.InitialVelocity);

var step = new SemiStepParams(
    options.Dt,
    options.SpringStiffness,
    options.SpringDamping,
    new Vector3(0f, options.GravityY, 0f),
    options.VelocityDamping,
    options.Substeps);

var stopwatch = Stopwatch.StartNew();
for (int i = 0; i < options.Steps; i++)
{
    system.StepSemi(step);
}
stopwatch.Stop();

var checksum = system.ComputeStateChecksum();
var elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
var stepsPerSecond = options.Steps / Math.Max(stopwatch.Elapsed.TotalSeconds, double.Epsilon);

Console.WriteLine($"cpu_mass_spring_benchmark");
Console.WriteLine($"particles={options.ParticleCount} springs={system.SpringCount} steps={options.Steps} substeps={options.Substeps}");
Console.WriteLine($"dt={options.Dt.ToString(CultureInfo.InvariantCulture)} stiffness={options.SpringStiffness.ToString(CultureInfo.InvariantCulture)} damping={options.SpringDamping.ToString(CultureInfo.InvariantCulture)}");
Console.WriteLine($"elapsed_ms={elapsedMs.ToString("F3", CultureInfo.InvariantCulture)} steps_per_sec={stepsPerSecond.ToString("F3", CultureInfo.InvariantCulture)}");
Console.WriteLine($"checksum={checksum.ToString("F6", CultureInfo.InvariantCulture)}");

readonly record struct SemiStepParams(
    float Dt,
    float SpringStiffness,
    float SpringDamping,
    Vector3 Gravity,
    float VelocityDamping,
    int Substeps);

readonly record struct SpringEdge(int A, int B, float RestLength);

sealed class CpuMassSpringSystem
{
    private readonly float[] _masses;
    private readonly bool[] _fixedMask;
    private readonly SpringEdge[] _springs;
    private readonly Vector3[] _forces;

    private CpuMassSpringSystem(
        Vector3[] positions,
        Vector3[] velocities,
        float[] masses,
        bool[] fixedMask,
        SpringEdge[] springs)
    {
        Positions = positions;
        Velocities = velocities;
        _masses = masses;
        _fixedMask = fixedMask;
        _springs = springs;
        _forces = new Vector3[positions.Length];
    }

    public Vector3[] Positions { get; }
    public Vector3[] Velocities { get; }
    public int ParticleCount => Positions.Length;
    public int SpringCount => _springs.Length;

    public static CpuMassSpringSystem CreateChain(int particleCount, float spacing, float mass, bool fixedFirstParticle)
    {
        if (particleCount < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(particleCount), "particleCount must be >= 2");
        }

        var positions = new Vector3[particleCount];
        var velocities = new Vector3[particleCount];
        var masses = new float[particleCount];
        var fixedMask = new bool[particleCount];
        var springs = new SpringEdge[particleCount - 1];

        for (int i = 0; i < particleCount; i++)
        {
            positions[i] = new Vector3(i * spacing, 0f, 0f);
            velocities[i] = Vector3.Zero;
            masses[i] = mass;
            fixedMask[i] = fixedFirstParticle && i == 0;

            if (i < particleCount - 1)
            {
                springs[i] = new SpringEdge(i, i + 1, spacing);
            }
        }

        return new CpuMassSpringSystem(positions, velocities, masses, fixedMask, springs);
    }

    public void SetInitialVelocity(Vector3 velocity)
    {
        for (int i = 0; i < Velocities.Length; i++)
        {
            if (_fixedMask[i])
            {
                continue;
            }

            Velocities[i] = velocity;
        }
    }

    public void StepSemi(in SemiStepParams step)
    {
        int substeps = Math.Max(step.Substeps, 1);
        float dtSub = step.Dt / substeps;

        for (int sub = 0; sub < substeps; sub++)
        {
            InitForces(step.Gravity);
            AccumulateSpringForces(step.SpringStiffness, step.SpringDamping);
            IntegrateSemi(dtSub, step.VelocityDamping);
        }
    }

    public float ComputeStateChecksum()
    {
        float sum = 0f;
        for (int i = 0; i < ParticleCount; i++)
        {
            var p = Positions[i];
            var v = Velocities[i];
            sum += p.X + p.Y + p.Z + v.X + v.Y + v.Z;
        }

        return sum;
    }

    private void InitForces(Vector3 gravity)
    {
        for (int i = 0; i < ParticleCount; i++)
        {
            _forces[i] = _fixedMask[i] ? Vector3.Zero : gravity * _masses[i];
        }
    }

    private void AccumulateSpringForces(float springStiffness, float springDamping)
    {
        for (int s = 0; s < _springs.Length; s++)
        {
            var edge = _springs[s];
            int a = edge.A;
            int b = edge.B;

            var pa = Positions[a];
            var pb = Positions[b];
            var va = Velocities[a];
            var vb = Velocities[b];

            var diff = pb - pa;
            float dist2 = Vector3.Dot(diff, diff);
            if (dist2 < Epsilon)
            {
                continue;
            }

            float invDist = 1.0f / MathF.Sqrt(dist2);
            float dist = dist2 * invDist;
            var dir = diff * invDist;
            var relVel = vb - va;

            float relAlong = Vector3.Dot(relVel, dir);
            float springForce = springStiffness * (dist - edge.RestLength);
            float dampingForce = springDamping * relAlong;
            var total = dir * (springForce + dampingForce);

            if (!_fixedMask[a])
            {
                _forces[a] += total;
            }

            if (!_fixedMask[b])
            {
                _forces[b] -= total;
            }
        }
    }

    private void IntegrateSemi(float dt, float velocityDamping)
    {
        for (int i = 0; i < ParticleCount; i++)
        {
            if (_fixedMask[i])
            {
                Velocities[i] = Vector3.Zero;
                continue;
            }

            float invMass = 1.0f / MathF.Max(_masses[i], Epsilon);
            var accel = _forces[i] * invMass;

            var v = Velocities[i] + accel * dt;
            var x = Positions[i] + v * dt;
            v *= velocityDamping;

            Velocities[i] = v;
            Positions[i] = x;
        }
    }
}

sealed class CliOptions
{
    public int ParticleCount { get; private set; } = 8192;
    public int Steps { get; private set; } = 200;
    public int Substeps { get; private set; } = 1;
    public float Dt { get; private set; } = 0.016f;
    public float SpringStiffness { get; private set; } = 120f;
    public float SpringDamping { get; private set; } = 0.2f;
    public float VelocityDamping { get; private set; } = 0.999f;
    public float GravityY { get; private set; } = -9.81f;
    public float Mass { get; private set; } = 1f;
    public float Spacing { get; private set; } = 0.1f;
    public bool FixedFirstParticle { get; private set; } = true;
    public Vector3 InitialVelocity { get; private set; } = new(0f, 0f, 0f);

    public static CliOptions Parse(string[] args)
    {
        var options = new CliOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            string value = i + 1 < args.Length ? args[i + 1] : string.Empty;

            switch (key)
            {
                case "--particles":
                    options.ParticleCount = ParseInt(value, key, min: 2);
                    i++;
                    break;
                case "--steps":
                    options.Steps = ParseInt(value, key, min: 1);
                    i++;
                    break;
                case "--substeps":
                    options.Substeps = ParseInt(value, key, min: 1);
                    i++;
                    break;
                case "--dt":
                    options.Dt = ParseFloat(value, key, minExclusive: 0f);
                    i++;
                    break;
                case "--stiffness":
                    options.SpringStiffness = ParseFloat(value, key, minInclusive: 0f);
                    i++;
                    break;
                case "--damping":
                    options.SpringDamping = ParseFloat(value, key, minInclusive: 0f);
                    i++;
                    break;
                case "--velocity-damping":
                    options.VelocityDamping = ParseFloat(value, key, minInclusive: 0f);
                    i++;
                    break;
                case "--gravity-y":
                    options.GravityY = ParseFloat(value, key);
                    i++;
                    break;
                case "--mass":
                    options.Mass = ParseFloat(value, key, minExclusive: 0f);
                    i++;
                    break;
                case "--spacing":
                    options.Spacing = ParseFloat(value, key, minExclusive: 0f);
                    i++;
                    break;
                case "--fixed-first":
                    options.FixedFirstParticle = ParseBool(value, key);
                    i++;
                    break;
                case "--initial-vx":
                {
                    var v = options.InitialVelocity;
                    options.InitialVelocity = new Vector3(ParseFloat(value, key), v.Y, v.Z);
                    i++;
                    break;
                }
                case "--initial-vy":
                {
                    var v = options.InitialVelocity;
                    options.InitialVelocity = new Vector3(v.X, ParseFloat(value, key), v.Z);
                    i++;
                    break;
                }
                case "--initial-vz":
                {
                    var v = options.InitialVelocity;
                    options.InitialVelocity = new Vector3(v.X, v.Y, ParseFloat(value, key));
                    i++;
                    break;
                }
                case "--help":
                case "-h":
                    PrintHelpAndExit();
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: {key}");
            }
        }

        return options;
    }

    private static int ParseInt(string text, string key, int min)
    {
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) || value < min)
        {
            throw new ArgumentException($"Invalid value for {key}: {text}");
        }

        return value;
    }

    private static float ParseFloat(string text, string key, float? minInclusive = null, float? minExclusive = null)
    {
        if (!float.TryParse(text, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float value))
        {
            throw new ArgumentException($"Invalid value for {key}: {text}");
        }

        if (minInclusive.HasValue && value < minInclusive.Value)
        {
            throw new ArgumentException($"Value for {key} must be >= {minInclusive.Value.ToString(CultureInfo.InvariantCulture)}");
        }

        if (minExclusive.HasValue && value <= minExclusive.Value)
        {
            throw new ArgumentException($"Value for {key} must be > {minExclusive.Value.ToString(CultureInfo.InvariantCulture)}");
        }

        return value;
    }

    private static bool ParseBool(string text, string key)
    {
        if (string.Equals(text, "true", StringComparison.OrdinalIgnoreCase) || text == "1")
        {
            return true;
        }

        if (string.Equals(text, "false", StringComparison.OrdinalIgnoreCase) || text == "0")
        {
            return false;
        }

        throw new ArgumentException($"Invalid value for {key}: {text} (expected true/false/1/0)");
    }

    private static void PrintHelpAndExit()
    {
        Console.WriteLine("CPU Spring-Mass-Damper benchmark options:");
        Console.WriteLine("  --particles <int>          Particle count (>=2)");
        Console.WriteLine("  --steps <int>              Number of simulation steps (>=1)");
        Console.WriteLine("  --substeps <int>           Semi-step substeps (>=1)");
        Console.WriteLine("  --dt <float>               Timestep in seconds (>0)");
        Console.WriteLine("  --stiffness <float>        Spring stiffness (>=0)");
        Console.WriteLine("  --damping <float>          Spring damping (>=0)");
        Console.WriteLine("  --velocity-damping <float> Global velocity damping (>=0)");
        Console.WriteLine("  --gravity-y <float>        Gravity Y");
        Console.WriteLine("  --mass <float>             Particle mass (>0)");
        Console.WriteLine("  --spacing <float>          Initial spacing (>0)");
        Console.WriteLine("  --fixed-first <bool>       Fix first particle (true/false/1/0)");
        Console.WriteLine("  --initial-vx <float>       Initial velocity x for free particles");
        Console.WriteLine("  --initial-vy <float>       Initial velocity y for free particles");
        Console.WriteLine("  --initial-vz <float>       Initial velocity z for free particles");
        Environment.Exit(0);
    }
}
