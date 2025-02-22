// document.addEventListener("DOMContentLoaded", () => {
//     const fileInput = document.querySelector(".upload-input");

//     fileInput.addEventListener("change", (event) => {
//         const file = event.target.files[0]; // Get the selected file
//         if (file) {
//             console.log("File selected:", file.name);
            
//             // Example: Show file name
//             alert(`You uploaded: ${file.name}`);

//             // Example: Process the file further
//             // Upload it to a server using fetch()
//             // processFile(file);
//         }
//     });
// });


// document.addEventListener("DOMContentLoaded", () => {
//     const fileInput = document.querySelector(".upload-input");
//     const uploadLabel = document.querySelector(".upload-btn");
//     const uploadForm = document.querySelector("form");

//     fileInput.addEventListener("change", (event) => {
//         const file = event.target.files[0]; // Get the selected file
//         if (file) {
//             console.log("File selected:", file.name);

//             // Change the button text to show file name
//             uploadLabel.textContent = `Selected: ${file.name}`;

//             // Show a success message
//             alert(`You uploaded: ${file.name}`);
//         }
//     });

//     uploadForm.addEventListener("submit", (event) => {
//         const file = fileInput.files[0];

//         if (!file) {
//             alert("Please select a file before submitting!");
//             event.preventDefault(); // Stop form submission
//             return;
//         }

//         // Show a loading message
//         uploadLabel.textContent = "Uploading...";
//     });
// });

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
        .then(response => response.text())
        .then(data => {
            console.log("Upload successful:", data);
            window.location.reload(); // Refresh the page to show uploaded files
        })
        .catch(error => {
            console.error("Upload failed:", error);
            alert("Error uploading file!");
        });
    });
});
