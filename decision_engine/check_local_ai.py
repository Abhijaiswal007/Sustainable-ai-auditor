from local_ai import parse_requirements_with_ai


test_request = (
    "I need at least 92% accuracy, "
    "latency below 150 milliseconds, "
    "and carbon below 60 kg."
)


print("Testing local Qwen3...")
print(f"Request: {test_request}")

constraints = parse_requirements_with_ai(
    test_request
)

print("\nExtracted constraints:")

for name, value in constraints.items():
    print(f"{name}: {value}")


assert constraints["min_accuracy"] == 0.92
assert constraints["max_latency_ms"] == 150
assert constraints["max_carbon_kgco2e"] == 60

print("\nLocal AI connection passed.")