async function predictTweet() {
    const tweet = document.getElementById("tweetInput").value;

    if (!tweet.trim()) {
        alert("Please enter a tweet.");
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: tweet })
        });

        const data = await response.json();

        document.getElementById("tweetText").innerText = data.tweet;
        document.getElementById("prediction").innerText = data.prediction;
        document.getElementById("confidence").innerText = data.confidence;

        document.getElementById("resultBox").classList.remove("hidden");
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to connect to backend.");
    }
}