document.addEventListener("DOMContentLoaded", function () {
  // Initialize Charts
  initializeDeliveryChart();
  initializeOrderChart();
  initializeMap();

  // Initialize Task Checkboxes
  initializeTasks();
});

// Delivery Performance Chart
function initializeDeliveryChart() {
  const ctx = document.getElementById("deliveryChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
      datasets: [
        {
          label: "On-time Deliveries",
          data: [65, 59, 80, 81, 56, 55, 40],
          borderColor: "#2ecc71",
          backgroundColor: "rgba(46, 204, 113, 0.1)",
          tension: 0.4,
          fill: true,
        },
        {
          label: "Delayed Deliveries",
          data: [28, 48, 40, 19, 86, 27, 90],
          borderColor: "#e74c3c",
          backgroundColor: "rgba(231, 76, 60, 0.1)",
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
    },
  });
}

// Order Distribution Chart
function initializeOrderChart() {
  const ctx = document.getElementById("orderChart").getContext("2d");

  // Create a container for the chart and legend
  const chartContainer = document.querySelector("#orderChart").parentElement;
  const legendContainer = document.createElement("div");
  legendContainer.className = "chart-legend";
  chartContainer.appendChild(legendContainer);

  // Define order categories with descriptions
  const orderCategories = [
    {
      name: "Perishable Goods",
      description: "Food items, flowers, and temperature-sensitive products",
      color: "#3498db",
    },
    {
      name: "Standard Packages",
      description: "Regular parcels and non-urgent deliveries",
      color: "#2ecc71",
    },
    {
      name: "Express Deliveries",
      description: "Time-critical and same-day delivery items",
      color: "#f1c40f",
    },
    {
      name: "Special Handling",
      description: "Fragile items, hazardous materials, and oversized packages",
      color: "#e74c3c",
    },
  ];

  // Create the chart
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: orderCategories.map((cat) => cat.name),
      datasets: [
        {
          data: [30, 40, 20, 10],
          backgroundColor: orderCategories.map((cat) => cat.color),
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false, // Hide default legend
        },
      },
      cutout: "70%",
    },
  });

  // Create custom legend with descriptions
  const legendHTML = orderCategories
    .map(
      (cat) => `
    <div class="legend-item">
      <div class="legend-color" style="background-color: ${cat.color}"></div>
      <div class="legend-text">
        <div class="legend-title">${cat.name}</div>
        <div class="legend-description">${cat.description}</div>
      </div>
    </div>
  `
    )
    .join("");

  legendContainer.innerHTML = legendHTML;

  // Add styles for the legend
  const style = document.createElement("style");
  style.textContent = `
    .chart-legend {
      margin-top: 1rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      padding: 1rem;
      background: #f8f9fa;
      border-radius: 8px;
    }
    .legend-item {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
    }
    .legend-color {
      width: 16px;
      height: 16px;
      border-radius: 4px;
      flex-shrink: 0;
      margin-top: 3px;
    }
    .legend-text {
      flex: 1;
    }
    .legend-title {
      font-weight: 500;
      color: #2c3e50;
      margin-bottom: 0.25rem;
    }
    .legend-description {
      font-size: 0.8rem;
      color: #7f8c8d;
      line-height: 1.4;
    }
  `;
  document.head.appendChild(style);
}

// Initialize Google Map
function initializeMap() {
  const map = new google.maps.Map(document.getElementById("deliveryMap"), {
    center: { lat: -1.2921, lng: 36.8219 }, // Nairobi coordinates
    zoom: 12,
    styles: [
      {
        featureType: "all",
        elementType: "geometry",
        stylers: [{ color: "#f5f5f5" }],
      },
      {
        featureType: "water",
        elementType: "geometry",
        stylers: [{ color: "#e9e9e9" }, { lightness: 17 }],
      },
    ],
  });

  // Add sample delivery markers
  const deliveries = [
    { lat: -1.2921, lng: 36.8219, title: "Delivery #1234" },
    { lat: -1.3021, lng: 36.8319, title: "Delivery #1235" },
    { lat: -1.2821, lng: 36.8119, title: "Delivery #1236" },
  ];

  deliveries.forEach((delivery) => {
    new google.maps.Marker({
      position: { lat: delivery.lat, lng: delivery.lng },
      map: map,
      title: delivery.title,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 8,
        fillColor: "#3498db",
        fillOpacity: 1,
        strokeColor: "#ffffff",
        strokeWeight: 2,
      },
    });
  });
}

// Task Management
function initializeTasks() {
  const taskCheckboxes = document.querySelectorAll(
    '.task-item input[type="checkbox"]'
  );

  taskCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", function () {
      const taskLabel = this.nextElementSibling;
      if (this.checked) {
        taskLabel.style.textDecoration = "line-through";
        taskLabel.style.color = "#95a5a6";
      } else {
        taskLabel.style.textDecoration = "none";
        taskLabel.style.color = "#2c3e50";
      }
    });
  });
}

// Real-time Updates Simulation
function simulateRealTimeUpdates() {
  setInterval(() => {
    // Simulate new activity
    const activities = [
      "New delivery assigned to Driver #1234",
      "Order #5678 has been completed",
      "Alert: Route deviation detected",
      "Driver #5678 has started their route",
      "Package #9012 has been delivered",
    ];

    const activityList = document.querySelector(".activity-list");
    const newActivity = document.createElement("div");
    newActivity.className = "activity-item";
    newActivity.innerHTML = `
            <div class="activity-icon">
                <i class="fas fa-truck"></i>
            </div>
            <div class="activity-details">
                <p class="activity-text">${
                  activities[Math.floor(Math.random() * activities.length)]
                }</p>
                <p class="activity-time">Just now</p>
            </div>
        `;

    activityList.insertBefore(newActivity, activityList.firstChild);
    if (activityList.children.length > 5) {
      activityList.removeChild(activityList.lastChild);
    }
  }, 30000); // Update every 30 seconds
}

// Start real-time updates
simulateRealTimeUpdates();
