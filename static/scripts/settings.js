document.addEventListener("DOMContentLoaded", function () {
  // Navigation handling
  const navItems = document.querySelectorAll(".nav-item");
  const sections = document.querySelectorAll(".settings-section");

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      // Remove active class from all items and sections
      navItems.forEach((nav) => nav.classList.remove("active"));
      sections.forEach((section) => section.classList.remove("active"));

      // Add active class to clicked item
      item.classList.add("active");

      // Show corresponding section
      const sectionId = item.getAttribute("data-section");
      document.getElementById(sectionId).classList.add("active");
    });
  });

  // Form submissions
  const forms = document.querySelectorAll(".settings-form");
  forms.forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const formData = new FormData(form);
      const data = Object.fromEntries(formData.entries());

      try {
        const response = await fetch(
          `/api/settings/${form.id.replace("-form", "")}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
          }
        );

        if (response.ok) {
          showNotification("Settings saved successfully!", "success");
        } else {
          const error = await response.json();
          showNotification(error.message || "Failed to save settings", "error");
        }
      } catch (error) {
        showNotification("An error occurred while saving settings", "error");
      }
    });
  });

  // Password validation
  const securityForm = document.getElementById("security-form");
  if (securityForm) {
    securityForm.addEventListener("submit", (e) => {
      const newPassword = document.getElementById("newPassword").value;
      const confirmPassword = document.getElementById("confirmPassword").value;

      if (newPassword && newPassword !== confirmPassword) {
        e.preventDefault();
        showNotification("Passwords do not match", "error");
      }
    });
  }

  // Theme switching
  const themeSelect = document.getElementById("theme");
  if (themeSelect) {
    themeSelect.addEventListener("change", (e) => {
      const theme = e.target.value;
      document.body.className = theme;
      localStorage.setItem("theme", theme);
    });
  }

  // Font size switching
  const fontSizeSelect = document.getElementById("fontSize");
  if (fontSizeSelect) {
    fontSizeSelect.addEventListener("change", (e) => {
      const size = e.target.value;
      document.body.style.fontSize =
        size === "small" ? "14px" : size === "large" ? "18px" : "16px";
      localStorage.setItem("fontSize", size);
    });
  }
});

// Notification system
function showNotification(message, type = "info") {
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;
  notification.textContent = message;

  // Add styles
  notification.style.position = "fixed";
  notification.style.top = "20px";
  notification.style.right = "20px";
  notification.style.padding = "1rem 2rem";
  notification.style.borderRadius = "4px";
  notification.style.color = "white";
  notification.style.zIndex = "1000";
  notification.style.animation = "slideIn 0.5s ease-out";

  // Set background color based on type
  switch (type) {
    case "success":
      notification.style.backgroundColor = "#2ecc71";
      break;
    case "error":
      notification.style.backgroundColor = "#e74c3c";
      break;
    default:
      notification.style.backgroundColor = "#3498db";
  }

  document.body.appendChild(notification);

  // Remove notification after 3 seconds
  setTimeout(() => {
    notification.style.animation = "slideOut 0.5s ease-in";
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 500);
  }, 3000);
}

// Add animation keyframes
const style = document.createElement("style");
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
