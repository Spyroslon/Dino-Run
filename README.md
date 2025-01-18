# Dino-Run
Reinforcement Learning on Chrome's Dino Run using **Playwright** and OpenAI's **Gymnasium** libraries.

The different approach from existing solutions is instead of using captured frames to just use the values stored in Javascript Objects.

### **Key Parameters**
- **Dino Status:** Current state of the Dino (e.g., running, jumping, crashed).  
- **Distance:** Total distance traveled.  
- **Dino Current Speed:** Current game speed.  
- **Dino yPos:** Dino's vertical position.  
- **List of Obstacles:** Details of upcoming obstacles, including their coordinates and sizes.  

---

### **Setting Up the Virtual Environment**
1. Create a virtual environment:  
   ```bash
   python -m venv .venv
   ```  
2. Activate the virtual environment:  
   - **Windows:**  
     ```bash
     .venv\Scripts\activate
     ```
   - **Linux/Mac:**  
     ```bash
     source .venv/bin/activate
     ```  
3. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Install Playwright and Chromium:  
   ```bash
   playwright install chromium
   ```

---

### **Tags**  
#### **Version 0.0.1**  
- Included all possible actions: `run`, `jump`, `fall`, `duck`, `stand`.  

#### **Version 1.0**  
- Reduced actions to `run` and `jump`.  
- Stable performance with `n_envs=1` but mediocre results.  

#### **Version 2.0**  
- Focused on `run` and `jump` actions only.  
- Incorporated significant fixes and tuning, but required extensive training. 

---

### **Version 2.0 Highlights**  

#### **Performance**  
- **Trained Model:** `ppo_4_dino_280000_steps.zip`  
- **Highest Score:** **378** (tested over 100 runs).  

#### **Fixes**  
- Tuned observation space bounds for better state representation.  
- Corrected the observation returned in each step.  
- Simplified the reward function to make it more intuitive.  
- Improved jump consistency by increasing key hold time during jumping.  

#### **Issues**  
- **Low FPS:**  
  - Increasing the sleep time during jumping reduced FPS significantly, impacting performance in fast-paced stages of the game.  
  - Training duration increased substantially due to the low FPS.  
- **Action Optimization:**  
  - The reduced FPS limited opportunities to perform optimal actions in later stages, where speed increases dramatically.  

---

### **Future Improvements**  
1. Expand the `action_space` to include **short jump** and **high jump** for finer control.  
2. Investigate alternative methods to make jumping more consistent while reducing the required sleep time to improve FPS.  
3. Reintroduce additional actions (`duck`, `stand`, `fall`) for a more comprehensive action set.  
4. Continue fine-tuning the existing trained model to achieve better performance and generalization.

---