document.addEventListener("DOMContentLoaded", () => {
    console.log("Page loaded. Checking for elements...");

    // ✅ 1️⃣ Handle File Upload Logic
    const fileInput = document.querySelector(".upload-input");
    if (fileInput) {
        console.log("Upload input found. Setting up event listener.");

        fileInput.addEventListener("change", (event) => {
            const file = event.target.files[0]; // Get the selected file
            if (!file) return; // Prevent empty submissions

            console.log("File selected:", file.name);

            // Create a FormData object to manually submit the file
            const formData = new FormData();
            formData.append("file", file); // Ensure the file is properly attached

            // Display "Uploading..." message
            fileInput.parentElement.textContent = `Uploading: ${file.name}...`;

            // Use fetch() to submit the form programmatically
            fetch("/", {
                method: "POST",
                body: formData
            })
            .then(response => {
                if (response.redirected) {
                    window.location.href = response.url; // ✅ Follow the redirect
                } else {
                    return response.text(); // ✅ If no redirect, log the response
                }
            })
            .catch(error => {
                console.error("Upload failed:", error);
                alert("Error uploading file!");
            });
        });
    } else {
        console.warn("No file input found. Skipping upload logic.");
    }

    // ✅ 2️⃣ Handle Heatmap Fetching (Only on /view/ pages)
    if (window.location.pathname.includes("/view/")) {
        console.log("Detected /view/ page. Fetching heatmap data...");
        fetchHeatmapData();
    }

    // ✅ Function to fetch heatmap data from Flask
    function fetchHeatmapData() {
        fetch("/get_model_output") // Flask route to fetch model output
            .then(response => response.json())
            .then(modelOutput => {
                console.log("Received model output:", modelOutput);
                updateHeatmap(modelOutput);
            })
            .catch(error => {
                console.error("Error fetching model output:", error);
            });
    }

    // ✅ Function to update heatmap colors
    function updateHeatmap(modelOutput) {
        function getHeatmapColor(intensity) {
            if (intensity <= 0.2) return "#00FF00"; // Green (Low strain)
            if (intensity <= 0.5) return "#FFFF00"; // Yellow (Medium strain)
            if (intensity <= 0.8) return "#FFA500"; // Orange (High strain)
            return "#FF0000"; // Red (Very High strain)
        }

        Object.keys(modelOutput).forEach((bodyPart) => {
            const partElement = document.getElementById(bodyPart);
            if (partElement) {
                partElement.setAttribute("fill", getHeatmapColor(modelOutput[bodyPart]));
                console.log(`Updated ${bodyPart} to ${getHeatmapColor(modelOutput[bodyPart])}`);
            } else {
                console.warn(`Element with ID '${bodyPart}' not found.`);
            }
        });
    }
});

