# Product Guidelines

## Development & Code Quality
- **Test-Driven Development (TDD):** All new features must be implemented with accompanying unit and integration tests to ensure reliability.
- **Comprehensive Docstrings:** Every module, class, and function must have thorough documentation (e.g., NumPy style) to assist scientific users and developers.
- **Scientific Python Best Practices:** Adhere to common scientific Python patterns and PEP 8 for consistency and readability.

## Performance & Architecture
- **Highly-Performant Codebase:** Prioritize speed and computational efficiency, especially when leveraging JAX and NumPy.
- **Modular Design:** Maintain a modular structure to ensure components are reusable, testable, and easily replaceable as new ML methods emerge.
- **Scalability-Ready:** Design core algorithms with future multi-GPU and HPC scalability in mind.

## Interface & Interaction
- **Library-First Design (API):** The primary way to use this tool is as a robust Python library, allowing researchers to integrate it into their own complex simulation workflows.
- **Developer-Friendly API:** Focus on a clean, intuitive, and well-documented API for ease of use by ML engineers and researchers.

## Scientific Rigor & Accuracy
- **Validation & Verification Suites:** Implement automated tests to verify the accuracy of predicted energy, forces, and other physical properties against reference data.
- **Strict Unit Management:** Use a consistent set of units (e.g., eV for energy, Angstrom for distance) throughout the entire codebase to prevent physical errors.
- **Physical Sanity Checks:** Ensure that energy conservation and other physical constraints are maintained during simulations.
