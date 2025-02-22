document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.querySelector(".upload-input");
    const uploadForm = document.querySelector(".upload-container");

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
        // .then(response => response.text())
        // .then(data => {
        //     console.log("Upload successful:", data);
        //     window.location.reload(); // Refresh the page to show uploaded files
        // })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url; // Follow the redirect
            } else {
                return response.text(); // If no redirect, log the response
            }
        })
        .catch(error => {
            console.error("Upload failed:", error);
            alert("Error uploading file!");
        });
    });
});

