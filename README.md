
# **StrAIN: Workout Strain Analyzer**

## **Overview**
This project analyzes user-uploaded workout videos to detect which areas of the body experience the most strain. Using pose estimation and biomechanical heuristics, it provides visual and textual feedback on potential injury risks and load management.

---

## **Input**
- Video (one person, full body)
- Weight (for Force calculation)
  
---

## **Output**
 - Newtons of Force on knees, hips, and ankles
 - Diagram of Strain Score on Body: Heatmap visualization

---
## **Dataset**

This project utilizes the **[GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics]** dataset, sourced from **[Boston University]**.  
The dataset provides synchronized ground reaction force (GRF) data alongside motion capture information. This dataset is valuable for biomechanical analysis, motion prediction, and human movement research.  

### Citation  

 [GroundLink Dataset on arXiv](https://arxiv.org/abs/2302.07879)  

---


## **Tech Stack**
- OpenCV (Computer Vision)
- MediaPipe (Pose Estimation & Motion Tracking)
- Flask (Web Framework)
- HTML, CSS (User Interface)
- JS (Interactive elements & client-side logic)
- Vercel (Deployment)

