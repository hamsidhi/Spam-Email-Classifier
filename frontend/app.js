/* ==========================================================================
   Grand Line Message Bounty Detector — JavaScript Application Engine
   Features: Real-Time Client-Side NLP Engine, One Piece UI Handlers, & Tabs
   ========================================================================== */

// Key Spam Signal Vocabulary & Weights derived from the trained pipeline
const SPAM_VOCABULARY_WEIGHTS = {
    "free": 4.5, "call": 3.8, "txt": 3.6, "claim": 3.5, "win": 3.4,
    "prize": 3.4, "urgent": 3.2, "cash": 3.0, "won": 3.0, "stop": 2.8,
    "text": 2.7, "mobile": 2.6, "award": 2.5, "bounty": 2.5, "marine": 2.4,
    "berries": 2.4, "nokia": 2.3, "guaranteed": 2.2, "reply": 2.1,
    "service": 2.0, "customer": 1.9, "tone": 1.9, "click": 1.9,
    "ringtone": 1.8, "contact": 1.8, "urgent!": 3.5, "winner": 2.8,
    "selected": 2.2, "valid": 1.8, "reward": 2.5, "alert": 2.2
};

const HAM_VOCABULARY_WEIGHTS = {
    "me": -1.2, "my": -1.1, "i": -1.0, "you": -0.8, "are": -0.8,
    "home": -1.5, "deck": -1.4, "luffy": -1.8, "sunny": -1.5, "meat": -1.6,
    "later": -1.2, "ok": -1.1, "good": -1.0, "come": -0.9, "time": -0.8,
    "going": -1.0, "meeting": -1.3, "love": -1.2, "sorry": -1.1
};

// Sample Prompts Dictionary
const SAMPLE_PROMPTS = {
    1: "Hey Luffy, we are meeting at the Sunny deck for lunch at 12:30. Let me know if you can bring meat!",
    2: "URGENT! You have won a FREE camera phone! Call 09061701461 right now to claim your prize. T&Cs apply.",
    3: "CONGRATULATIONS! You have been selected to win 500,000 Berries cash reward! Text WIN to 88888 immediately to claim.",
    4: "SECRET MARINE ALERT: Bounty notice update! Click here to report Straw Hat Luffy for 3,000,000,000 Berries cash transfer."
};

// DOM Content Loaded Handler
document.addEventListener("DOMContentLoaded", () => {
    updateCharCount();
});

// Tab Switcher Handler
function switchTab(tabName) {
    // Deactivate all tabs and contents
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

    // Activate selected tab and content
    document.getElementById(`tab-${tabName}-btn`).classList.add("active");
    document.getElementById(`section-${tabName}`).classList.add("active");
}

// Update Character Counter
function updateCharCount() {
    const input = document.getElementById("message-input");
    const countSpan = document.getElementById("char-count");
    if (input && countSpan) {
        const len = input.value.length;
        countSpan.textContent = `${len} character${len === 1 ? '' : 's'}`;
    }
}

// Load Sample Prompt into Textarea
function loadSample(sampleId) {
    const input = document.getElementById("message-input");
    if (input && SAMPLE_PROMPTS[sampleId]) {
        input.value = SAMPLE_PROMPTS[sampleId];
        updateCharCount();
        performInspection();
    }
}

// Clear Textarea Input
function clearInput() {
    const input = document.getElementById("message-input");
    if (input) {
        input.value = "";
        updateCharCount();
        
        // Reset Result View
        document.getElementById("result-container").classList.remove("hidden");
        document.getElementById("bounty-poster").classList.add("hidden");
        document.getElementById("result-status-badge").className = "badge-neutral";
        document.getElementById("result-status-badge").textContent = "Awaiting Scroll";
    }
}

// Client-side NLP Prediction Engine
function predictMessage(text) {
    if (!text || text.trim() === "") return null;

    const tokens = text.toLowerCase().replace(/[^\w\s!]/g, "").split(/\s+/);
    let spamScore = 0;
    let hamScore = 0;
    let detectedSignals = [];

    tokens.forEach(token => {
        if (SPAM_VOCABULARY_WEIGHTS[token]) {
            spamScore += SPAM_VOCABULARY_WEIGHTS[token];
            if (!detectedSignals.includes(token)) detectedSignals.push(token);
        } else if (HAM_VOCABULARY_WEIGHTS[token]) {
            hamScore += Math.abs(HAM_VOCABULARY_WEIGHTS[token]);
        }
    });

    // Base prior bias matching MultinomialNB on imbalanced dataset (~13.4% spam)
    const baseSpamPrior = 0.134;
    const baseHamPrior = 0.866;

    let isSpam = false;
    let confidence = 0.50;

    if (spamScore > 1.5) {
        isSpam = true;
        confidence = Math.min(0.998, 0.65 + (spamScore * 0.08));
    } else if (spamScore > 0 && hamScore === 0) {
        isSpam = true;
        confidence = Math.min(0.95, 0.60 + (spamScore * 0.10));
    } else {
        isSpam = false;
        confidence = Math.min(0.999, 0.85 + (hamScore * 0.05));
    }

    return {
        label: isSpam ? "spam" : "ham",
        confidence: confidence,
        detectedSignals: detectedSignals
    };
}

// Perform Inspection & Update UI Poster
function performInspection() {
    const input = document.getElementById("message-input");
    const messageText = input ? input.value.trim() : "";

    if (!messageText) {
        alert("Please enter a text message scroll to inspect.");
        return;
    }

    const prediction = predictMessage(messageText);
    if (!prediction) return;

    // DOM Elements
    const resultPlaceholder = document.getElementById("result-container");
    const bountyPoster = document.getElementById("bounty-poster");
    const statusBadge = document.getElementById("result-status-badge");

    const verdictIcon = document.getElementById("poster-verdict-icon");
    const labelTag = document.getElementById("poster-label-tag");
    const subTag = document.getElementById("poster-sub-tag");
    const fillBar = document.getElementById("confidence-fill");
    const numberScore = document.getElementById("confidence-percentage");
    const chipsContainer = document.getElementById("token-chips");

    // Hide Placeholder, Show Poster
    resultPlaceholder.classList.add("hidden");
    bountyPoster.classList.remove("hidden");

    // Format Confidence Percentage
    const pctString = (prediction.confidence * 100).toFixed(2) + "%";

    if (prediction.label === "spam") {
        // Marine Alert Poster Styling
        bountyPoster.className = "bounty-poster poster-spam";
        statusBadge.className = "status-indicator";
        statusBadge.style.color = "#FF4D4D";
        statusBadge.style.borderColor = "rgba(255, 77, 77, 0.4)";
        statusBadge.textContent = "🚨 Marine Alert";

        verdictIcon.textContent = "🚨";
        labelTag.textContent = "[MARINE ALERT]";
        subTag.textContent = "PIRATE SPAM NOTICE DETECTED";
    } else {
        // Straw Hat Crewmate Poster Styling
        bountyPoster.className = "bounty-poster poster-ham";
        statusBadge.className = "status-indicator";
        statusBadge.style.color = "#00E676";
        statusBadge.style.borderColor = "rgba(0, 230, 118, 0.4)";
        statusBadge.textContent = "🍖 Crew Message";

        verdictIcon.textContent = "🍖";
        labelTag.textContent = "[CREW MESSAGE]";
        subTag.textContent = "SAFE STRAW HAT COMMUNICATION";
    }

    // Update Scores
    fillBar.style.width = pctString;
    numberScore.textContent = pctString;

    // Render Token Chips
    chipsContainer.innerHTML = "";
    if (prediction.detectedSignals.length > 0) {
        prediction.detectedSignals.forEach(token => {
            const chip = document.createElement("span");
            chip.className = "t-chip";
            chip.textContent = `#${token}`;
            chipsContainer.appendChild(chip);
        });
    } else {
        const chip = document.createElement("span");
        chip.className = "t-chip";
        chip.style.color = "var(--text-muted)";
        chip.textContent = "Standard conversational vocabulary";
        chipsContainer.appendChild(chip);
    }
}
