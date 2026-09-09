const API_BASE_URL = "http://127.0.0.1:8000/api";


export async function uploadDataset(file) {
    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch(
        `${API_BASE_URL}/datasets/upload`,
        {
            method: "POST",
            body: formData
        }
    );

    if (!response.ok) {
        const error = await response.json();

        throw new Error(
            error.detail ||
            "Dataset upload failed."
        );
    }

    return response.json();
}


export async function runPipeline(
    storedFilename
) {
    const response = await fetch(
        `${API_BASE_URL}/pipeline/run`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                stored_filename: storedFilename
            })
        }
    );

    if (!response.ok) {
        const error = await response.json();

        throw new Error(
            error.detail ||
            "Pipeline execution failed."
        );
    }

    return response.json();
}


export async function fetchDatasetColumns(storedFilename) {
    const response = await fetch(
        `${API_BASE_URL}/datasets/${storedFilename}/columns`
    );

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fetch dataset columns.");
    }

    return response.json();
}


export async function analyzeEDA(storedFilename, targetColumn = null) {
    const response = await fetch(
        `${API_BASE_URL}/eda/analyze`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                stored_filename: storedFilename,
                target_column: targetColumn ? targetColumn : null
            })
        }
    );

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "EDA analysis failed.");
    }

    return response.json();
}


export async function getAIAgentRecommendation(storedFilename, targetColumn = null) {
    const response = await fetch(
        `${API_BASE_URL}/agent/recommend`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                stored_filename: storedFilename,
                target_column: targetColumn ? targetColumn : null
            })
        }
    );

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "AI Agent recommendation failed.");
    }

    return response.json();
}
