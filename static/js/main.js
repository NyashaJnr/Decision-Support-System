// Theme handling
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

// Font size handling
function applyFontSize(size) {
  document.body.className = document.body.className.replace(
    /font-size-\w+/,
    ""
  );
  document.body.classList.add(`font-size-${size}`);
  localStorage.setItem("fontSize", size);
}

// Compact view handling
function applyCompactView(compact) {
  if (compact) {
    document.body.classList.add("compact-view");
  } else {
    document.body.classList.remove("compact-view");
  }
  localStorage.setItem("compactView", compact);
}

// Initialize settings from localStorage
document.addEventListener("DOMContentLoaded", function () {
  // Apply theme
  const savedTheme = localStorage.getItem("theme") || "light";
  applyTheme(savedTheme);

  // Apply font size
  const savedFontSize = localStorage.getItem("fontSize") || "medium";
  applyFontSize(savedFontSize);

  // Apply compact view
  const savedCompactView = localStorage.getItem("compactView") === "true";
  applyCompactView(savedCompactView);
});

// Handle form submissions
document.addEventListener("submit", function (e) {
  if (e.target.matches("form")) {
    const submitButton = e.target.querySelector('button[type="submit"]');
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
    }
  }
});
