using System.Diagnostics;
using System.Globalization;

var options = CliOptions.Parse(args);

var stopwatch = Stopwatch.StartNew();
float position = options.Position;
float velocity = options.Velocity;

for (int i = 0; i < options.Steps; i++)
{
    OneDIntegrator.StepImplicit(ref position, ref velocity, options.Dt, options.Mass, options.Stiffness, options.Damping);
}
stopwatch.Stop();

float checksum = position + velocity;
var elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
var stepsPerSecond = options.Steps / Math.Max(stopwatch.Elapsed.TotalSeconds, double.Epsilon);

Console.WriteLine("mass_spring_benchmark");
Console.WriteLine("backend=csharp");
Console.WriteLine($"steps={options.Steps}");
Console.WriteLine($"position={position.ToString("F6", CultureInfo.InvariantCulture)} velocity={velocity.ToString("F6", CultureInfo.InvariantCulture)}");
Console.WriteLine($"dt={options.Dt.ToString(CultureInfo.InvariantCulture)} mass={options.Mass.ToString(CultureInfo.InvariantCulture)} stiffness={options.Stiffness.ToString(CultureInfo.InvariantCulture)} damping={options.Damping.ToString(CultureInfo.InvariantCulture)}");
Console.WriteLine($"elapsed_ms={elapsedMs.ToString("F3", CultureInfo.InvariantCulture)} steps_per_sec={stepsPerSecond.ToString("F3", CultureInfo.InvariantCulture)}");
Console.WriteLine($"output_x={position.ToString("F6", CultureInfo.InvariantCulture)} output_v={velocity.ToString("F6", CultureInfo.InvariantCulture)}");
Console.WriteLine($"checksum={checksum.ToString("F6", CultureInfo.InvariantCulture)}");

static class OneDIntegrator
{
    public static void StepImplicit(
        ref float position,
        ref float velocity,
        float dt,
        float mass,
        float stiffness,
        float damping)
    {
        float clampedDt = MathF.Max(dt, 1e-6f);
        float clampedMass = MathF.Max(mass, 1e-6f);

        float denom = 1.0f + (damping * clampedDt) / clampedMass + (stiffness * clampedDt * clampedDt) / clampedMass;
        if (MathF.Abs(denom) < 1e-6f)
        {
            denom = denom >= 0f ? 1e-6f : -1e-6f;
        }

        float newVelocity = (velocity - (stiffness * clampedDt / clampedMass) * position) / denom;
        float newPosition = position + clampedDt * newVelocity;
        velocity = newVelocity;
        position = newPosition;
    }
}

sealed class CliOptions
{
    public float Position { get; private set; } = 1f;
    public float Velocity { get; private set; } = 0f;
    public float Dt { get; private set; } = 0.016f;
    public float Mass { get; private set; } = 1f;
    public float Stiffness { get; private set; } = 120f;
    public float Damping { get; private set; } = 0.2f;
    public int Steps { get; private set; } = 200;

    public static CliOptions Parse(string[] args)
    {
        var options = new CliOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            string value = i + 1 < args.Length ? args[i + 1] : string.Empty;

            switch (key)
            {
                case "--position":
                    options.Position = ParseFloat(value, key);
                    i++;
                    break;
                case "--velocity":
                    options.Velocity = ParseFloat(value, key);
                    i++;
                    break;
                case "--dt":
                    options.Dt = ParseFloat(value, key, minExclusive: 0f);
                    i++;
                    break;
                case "--mass":
                    options.Mass = ParseFloat(value, key, minExclusive: 0f);
                    i++;
                    break;
                case "--stiffness":
                    options.Stiffness = ParseFloat(value, key, minInclusive: 0f);
                    i++;
                    break;
                case "--damping":
                    options.Damping = ParseFloat(value, key, minInclusive: 0f);
                    i++;
                    break;
                case "--steps":
                    options.Steps = ParseInt(value, key, min: 1);
                    i++;
                    break;
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

    private static void PrintHelpAndExit()
    {
        Console.WriteLine("C# 1D Spring-Mass-Damper benchmark options:");
        Console.WriteLine("  --position <float>   Initial position");
        Console.WriteLine("  --velocity <float>   Initial velocity");
        Console.WriteLine("  --dt <float>         Time-step (>0)");
        Console.WriteLine("  --mass <float>       Mass (>0)");
        Console.WriteLine("  --stiffness <float>  Spring stiffness (>=0)");
        Console.WriteLine("  --damping <float>    Spring damping (>=0)");
        Console.WriteLine("  --steps <int>        Number of integration steps (>=1)");
        Environment.Exit(0);
    }
}
