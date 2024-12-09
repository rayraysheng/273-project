// ocrScript.js

function analyzeFile() {
    var fileInput = document.getElementById("inputFile");
    var file = fileInput.files[0];

    if (file) {
        // Show the loading spinner
        document.getElementById("loadingIcon").style.display = "block";

        // Check file type
        if (file.type === "application/pdf") {
            // Handle PDF files
            handlePdf(file);
        } else if (file.type.startsWith("image/")) {
            // Handle Image files
            handleImage(file);
        } else {
            alert("Please select a valid image or PDF file.");
        }
    } else {
        alert("Please select a file.");
    }
}

function handleImage(imageFile) {
    var reader = new FileReader();
    reader.onload = function(event) {
        var image = new Image();
        image.src = event.target.result;
        document.getElementById("sourceImage").src = image.src;

        // Perform preprocessing if needed (adjust size, convert to grayscale, etc.)
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        image.onload = function() {
            // Resize the image for better OCR performance (if necessary)
            var width = image.width > 800 ? 800 : image.width;
            var height = (width / image.width) * image.height;
            canvas.width = width;
            canvas.height = height;

            ctx.drawImage(image, 0, 0, width, height);

            // Convert image to grayscale (optional but helps for OCR accuracy)
            var imageData = ctx.getImageData(0, 0, width, height);
            var data = imageData.data;
            for (var i = 0; i < data.length; i += 4) {
                var r = data[i];
                var g = data[i + 1];
                var b = data[i + 2];
                // Apply grayscale formula
                var avg = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                data[i] = data[i + 1] = data[i + 2] = avg;
            }
            ctx.putImageData(imageData, 0, 0);

            // Now perform OCR with the preprocessed image
            Tesseract.recognize(
                canvas.toDataURL(),
                'eng',
                {
                    logger: (m) => console.log(m),
                    oem: 1 // Use the default OCR engine mode (LSTM-based)
                }
            ).then(({ data: { text } }) => {
                document.getElementById("outputTextBox").value = text;

                // Hide the loading spinner when OCR is complete
                document.getElementById("loadingIcon").style.display = "none";
            }).catch((error) => {
                alert("Error processing image: " + error);

                // Hide the loading spinner in case of error
                document.getElementById("loadingIcon").style.display = "none";
            });
        };
    };
    reader.readAsDataURL(imageFile);
}

function handlePdf(pdfFile) {
    var reader = new FileReader();
    reader.onload = function(event) {
        var pdfData = new Uint8Array(event.target.result);

        // Use pdf.js to extract the first page as an image with higher resolution
        pdfjsLib.getDocument(pdfData).promise.then(function(pdf) {
            pdf.getPage(1).then(function(page) {
                var viewport = page.getViewport({ scale: 2.5 }); // Increase scale for better resolution
                var canvas = document.createElement('canvas');
                var context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                page.render({ canvasContext: context, viewport: viewport }).promise.then(function() {
                    var imgData = canvas.toDataURL();

                    // Show the PDF image in the UI
                    document.getElementById("sourceImage").src = imgData;

                    // Perform OCR on the extracted image from the PDF
                    Tesseract.recognize(
                        imgData,
                        'eng',
                        {
                            logger: (m) => console.log(m)
                        }
                    ).then(({ data: { text } }) => {
                        document.getElementById("outputTextBox").value = text;

                        // Hide the loading spinner when OCR is complete
                        document.getElementById("loadingIcon").style.display = "none";
                    }).catch((error) => {
                        alert("Error processing PDF: " + error);

                        // Hide the loading spinner in case of error
                        document.getElementById("loadingIcon").style.display = "none";
                    });
                });
            });
        });
    };
    reader.readAsArrayBuffer(pdfFile);
}
