
# **StrAIN: Workout Strain Analyzer**

## **Overview**
This project analyzes user-uploaded workout videos to detect which areas of the body experience the most strain. Using pose estimation and biomechanical heuristics, it provides visual and textual feedback on potential injury risks and load management.

---

## **Input**
- Video (Max 5 seconds, Full Body)
  
---

## **Output**
 - Strain Score: Overall intensity of strain  
 - Areas Most Affected: Key joints experiencing high stress
 - Diagram of Strain Score on Body: Heatmap visualization  

---

## **Tech Stack**
- **Backend:** Python, OpenCV, MediaPipe  
- **Frontend:** Flask, HTML/CSS  
