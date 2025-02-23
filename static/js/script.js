document.addEventListener("DOMContentLoaded", () => {
    console.log("✅ Page loaded. Initializing event listeners...");

    // Check if we are on the index page
    if (window.location.pathname === "/") {
        console.log("✅ On the upload page, initializing file upload logic...");

        /** ✅ 1️⃣ File Upload Logic **/
        const fileInput = document.querySelector("#file");
        const weightInput = document.querySelector("#weight");
        const uploadButton = document.querySelector("#uploadButton");

        // Log to verify the elements
        console.log("fileInput:", fileInput);
        console.log("weightInput:", weightInput);
        console.log("uploadButton:", uploadButton);

        // Ensure the elements exist before interacting with them
        if (!fileInput || !weightInput || !uploadButton) {
            console.error("❌ Required elements not found in the DOM!");
            return;  // Exit the script if the elements are not found
        }

        // ✅ Error message for missing weight
        const errorMessage = document.createElement("p");
        errorMessage.style.color = "red";
        errorMessage.style.display = "none";
        errorMessage.textContent = "Please enter your weight before uploading.";
        if (weightInput) {
            weightInput.parentElement.appendChild(errorMessage);
        }

        // ✅ Disable upload button until weight is entered
        uploadButton.disabled = true;
        weightInput.addEventListener("input", () => {
            if (weightInput.value.trim() !== "") {
                uploadButton.disabled = false;  // Enable button
                errorMessage.style.display = "none";  // Hide error message
            } else {
                uploadButton.disabled = true;  // Disable button
            }
        });

        // ✅ Open file dialog when clicking Upload Video
        uploadButton.addEventListener("click", () => {
            if (!weightInput.value.trim()) {
                alert("⚠️ Please enter your weight before uploading.");
                errorMessage.style.display = "block";  // Show error message
                return;
            }
            fileInput.click();  // Opens file dialog
        });

        // ✅ Handle File Selection & Upload
        fileInput.addEventListener("change", async () => {
            const weightValue = weightInput.value.trim();
            const file = fileInput.files[0];

            if (!file) {
                alert("⚠️ Please select a video file.");
                return;
            }

            console.log("📤 Uploading:", file.name, "with weight:", weightValue);

            const formData = new FormData();
            formData.append("file", file);
            formData.append("weight", weightValue);  // Include weight

            // Disable button & show "Uploading..." message
            uploadButton.textContent = `Uploading: ${file.name}...`;

            try {
                const response = await fetch("/", {
                    method: "POST",
                    body: formData
                });

                if (response.redirected) {
                    window.location.href = response.url; // ✅ Redirect on success
                } else {
                    const result = await response.text();
                    console.log("✅ Server Response:", result);
                }
            } catch (error) {
                console.error("❌ Upload failed:", error);
                alert("⚠️ Error uploading file!");
            } finally {
                uploadButton.textContent = "Upload Video";
                uploadButton.disabled = false;
            }
        });
    }

    /** ✅ 2️⃣ Handle Heatmap Fetching (Only on /view/ pages) **/
    if (window.location.pathname.includes("/view/")) {
        console.log("🔍 Detected /view/ page. Fetching heatmap data...");
        fetchHeatmapData();
    }

    /** ✅ Function to fetch heatmap data from Flask **/
    function fetchHeatmapData() {
        fetch("/get_model_output")
            .then(response => response.json())
            .then(modelOutput => {
                console.log("📊 Received model output:", modelOutput);
                updateHeatmap(modelOutput);
            })
            .catch(error => {
                console.error("❌ Error fetching model output:", error);
            });
    }

    function updateHeatmap(modelOutput) {
        console.log("🔵 Updating Heatmap...");
    
        Object.keys(modelOutput).forEach((bodyPart) => {
            const partElement = document.getElementById(bodyPart);
    
            if (partElement) {
                // Access the 'color' property from the object
                const color = modelOutput[bodyPart]?.color;  // ✅ Get color from API
                if (color) {
                    partElement.setAttributeNS(null, "fill", color);  // Apply the color to the SVG circle
    
                    console.log(`✅ Updated ${bodyPart} to ${color}`);
                } else {
                    console.warn(`⚠️ No color provided for '${bodyPart}'`);
                }
            } else {
                console.error(`❌ Element with ID '${bodyPart}' not found.`);
            }
        });
    }
});
