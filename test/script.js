function setMood(mood) {
  const mouth = document.getElementById("mouth");
  if (mood === "happy") {
    mouth.className = "mouth happy";
  } else if (mood === "sad") {
    mouth.className = "mouth sad";
  }
}
