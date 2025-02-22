document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.querySelector(".upload-input");

    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0]; // Get the selected file
        if (file) {
            console.log("File selected:", file.name);
            
            // Example: Show file name
            alert(`You uploaded: ${file.name}`);

            // Example: Process the file further
            // Upload it to a server using fetch()
            // processFile(file);
        }
    });
});
