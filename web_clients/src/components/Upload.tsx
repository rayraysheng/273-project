import React, { useState, useRef, ChangeEvent, DragEvent } from "react";
import { Upload, X, File, Send } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Input } from "./ui/input";

interface FileData {
  file: File;
  id: string;
  progress: number;
  status: "pending" | "uploading" | "completed" | "error";
  error: string | null;
}

const PDFUploader: React.FC = () => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [files, setFiles] = useState<FileData[]>([]);
  const [error, setError] = useState<string>("");
  const [title, setTitle] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (): void => {
    setIsDragging(false);
  };

  const validateFile = (file: File): boolean => {
    if (!file) return false;

    if (file.type !== "application/pdf") {
      setError("Please upload only PDF files");
      return false;
    }

    return true;
  };

  const handleFiles = (newFiles: FileList | null): void => {
    if (!newFiles) return;

    setError("");

    const validFiles = Array.from(newFiles)
      .filter(validateFile)
      .map((file) => ({
        file,
        id: Math.random().toString(36).substring(7),
        progress: 0,
        status: "pending" as const,
        error: null,
      }));

    if (validFiles.length > 0) {
      setFiles((prev) => [...prev, ...validFiles]);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>): void => {
    handleFiles(e.target.files);
    e.target.value = "";
  };

  const handleSubmit = async () => {
    if (files.length === 0) {
      setError("Please select files to upload");
      return;
    }

    setIsUploading(true);
    setError("");

    // Create a single FormData containing all files
    const formData = new FormData();
    formData.append("title", title);

    files.forEach((fileData) => {
      // This exactly matches FastAPI's expectation where files: List[UploadFile]
      formData.append("files", fileData.file);
    });

    try {
      const response = await fetch("http://localhost:8001/upload", {
        method: "POST",
        // Don't set Content-Type header - browser will set it automatically
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const result = await response.json();
      // Simulating upload progress for all files
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setFiles((prev) =>
          prev.map((f) => ({
            ...f,
            progress,
            status: progress === 100 ? "completed" : "uploading",
          }))
        );
      }
    } catch (err) {
      setError("Failed to upload files");
      setFiles((prev) =>
        prev.map((f) => ({
          ...f,
          status: "error",
          error: "Upload failed",
        }))
      );
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (fileId: string): void => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const getStatusColor = (status: FileData["status"]): string => {
    switch (status) {
      case "completed":
        return "text-green-500";
      case "error":
        return "text-red-500";
      default:
        return "text-gray-500";
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-4">
        <Input
          placeholder="Enter title for this upload"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          disabled={isUploading}
        />
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center ${
          isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300"
        } ${error ? "border-red-500" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept=".pdf"
          multiple
          onChange={handleFileInput}
        />

        <div className="space-y-4">
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <div>
            <p className="text-lg font-medium">Drop your PDF files here</p>
            <p className="text-sm text-gray-500">or</p>
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              className="mt-2"
            >
              Browse Files
            </Button>
          </div>
          <p className="text-sm text-gray-500">
            Upload your PDF files to get started
          </p>
        </div>
      </div>

      {files.length > 0 && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-4">
            {files.map((fileData) => (
              <div key={fileData.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <File
                      className={`h-4 w-4 ${getStatusColor(fileData.status)}`}
                    />
                    <span className="text-sm font-medium truncate">
                      {fileData.file.name}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(fileData.id)}
                    disabled={isUploading}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <Progress value={fileData.progress} className="w-full" />
                <div className="flex justify-between mt-2">
                  <span className="text-sm text-gray-500">
                    {fileData.status === "completed"
                      ? "Upload complete!"
                      : fileData.status === "error"
                      ? fileData.error
                      : fileData.progress > 0
                      ? `Uploading: ${fileData.progress}%`
                      : "Ready to upload"}
                  </span>
                  <span className="text-sm text-gray-500">
                    {(fileData.file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 flex justify-end">
            <Button
              onClick={handleSubmit}
              disabled={isUploading || files.length === 0}
              className="px-6"
            >
              <Send className="h-4 w-4 mr-2" />
              {isUploading ? "Uploading..." : "Upload Files"}
            </Button>
          </div>
        </>
      )}

      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

const UploadPage: React.FC = () => {
  const [showUploader, setShowUploader] = useState(false);

  return (
    <div className="p-8">
      <Button onClick={() => setShowUploader(!showUploader)} className="mb-6">
        {showUploader ? "Done" : "Upload Manuals"}
      </Button>

      {showUploader && <PDFUploader />}
    </div>
  );
};

export default UploadPage;
