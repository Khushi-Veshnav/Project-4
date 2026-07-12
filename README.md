Retail Sales Analytics & Demand Forecasting System
A comprehensive, data-driven analytics and forecasting solution designed to optimize retail supply chain operations, predict seasonal demand fluctuations, and minimize regional inventory inefficiencies.

This project encompasses everything from deep Exploratory Data Analysis (EDA) and predictive modeling to a production-ready dashboard that makes forecasting insights accessible to business executives.

🚀 Features
Exploratory Data Analysis (EDA): Deep-dive analysis into seasonal trends, regional growth drivers, and logistical shipping performance.

Predictive Demand Forecasting: A time-series forecasting engine built to project sales targets alongside conservative and aggressive confidence ranges.

Anomaly Detection: Algorithmic identification of historical revenue disruptions, helping the business isolate market shocks from operational failures.

Product Segmentation: Classification of inventory items based on revenue impact and demand stability to drive an optimized stocking strategy.

Interactive Web Application: A clean, user-friendly interface built to present raw data insights in plain, executive-friendly language.

📂 Project Structure
Plaintext
├── train.csv           # Historical retail transaction data
├── analysis.ipynb      # End-to-end data analysis, EDA, and model building
├── app.py              # Web application code for the interactive dashboard
├── requirements.txt    # Python dependencies required to run the project
└── Summary.docx        # Final Executive Business Report tailored for the CFO & Head of Supply Chain
🛠️ Installation & Setup
Follow these steps to set up and run the project locally on your machine.

1. Clone the Repository
Bash
git clone https://github.com/Khushi-Veshnav/Project-4.git
cd Project-4
2. Set Up a Virtual Environment (Recommended)
Bash
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Install all the required Python libraries using the requirements.txt file:

Bash
pip install -r requirements.txt
4. Run the Application
Launch the web interface locally:

Bash
python app.py
Once started, copy the local URL (usually [http://127.0.0.1:5000](http://127.0.0.1:5000) or similar) provided in your terminal and paste it into your web browser.

📊 Key Business Insights (Summary)
Regional Engines: The West and East regions serve as the primary growth drivers for the organization, showing stable, compounding revenue increases year-over-year.

The "March Surge": Historical data confirms a rigid annual demand spike in March (bouncing back strongly from low post-holiday volumes in January and February), requiring a proactive pre-season inventory build.

Inventory Optimization: High-revenue "Core Anchors" (such as the Canon imageCLASS copiers) have been isolated for Just-in-Time (JIT) replenishment to maximize working capital efficiency without risking stockouts.

For the full deep-dive analysis and data-backed recommendations, please refer to the Summary.docx file.
