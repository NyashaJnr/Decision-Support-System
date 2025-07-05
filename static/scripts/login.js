document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.querySelector(".login-form");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const roleSelect = document.getElementById("role");

  // Function to validate email format
  function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  // Function to show error message
  function showError(input, message) {
    const formGroup = input.closest(".form-group");
    const errorDiv =
      formGroup.querySelector(".error-message") ||
      document.createElement("div");
    errorDiv.className = "error-message text-danger mt-1";
    errorDiv.textContent = message;

    if (!formGroup.querySelector(".error-message")) {
      formGroup.appendChild(errorDiv);
    }

    input.classList.add("is-invalid");
  }

  // Function to clear error message
  function clearError(input) {
    const formGroup = input.closest(".form-group");
    const errorDiv = formGroup.querySelector(".error-message");
    if (errorDiv) {
      errorDiv.remove();
    }
    input.classList.remove("is-invalid");
  }

  // Add input event listeners for real-time validation
  emailInput.addEventListener("input", function () {
    clearError(this);
    if (this.value && !isValidEmail(this.value)) {
      showError(this, "Please enter a valid email address");
    }
  });

  passwordInput.addEventListener("input", function () {
    clearError(this);
    if (this.value && this.value.length < 6) {
      showError(this, "Password must be at least 6 characters long");
    }
  });

  roleSelect.addEventListener("change", function () {
    clearError(this);
    if (!this.value) {
      showError(this, "Please select a role");
    }
  });

  // Form submission validation
  loginForm.addEventListener("submit", function (e) {
    let isValid = true;

    // Validate email
    if (!emailInput.value) {
      showError(emailInput, "Email is required");
      isValid = false;
    } else if (!isValidEmail(emailInput.value)) {
      showError(emailInput, "Please enter a valid email address");
      isValid = false;
    }

    // Validate password
    if (!passwordInput.value) {
      showError(passwordInput, "Password is required");
      isValid = false;
    } else if (passwordInput.value.length < 6) {
      showError(passwordInput, "Password must be at least 6 characters long");
      isValid = false;
    }

    // Validate role
    if (!roleSelect.value) {
      showError(roleSelect, "Please select a role");
      isValid = false;
    }

    if (!isValid) {
      e.preventDefault();
    }
  });
});
