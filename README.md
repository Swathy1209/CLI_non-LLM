📊 Agentic Multi-Source Retrieval System (No LLM)
🧠 Overview

This project is a CLI-based Agentic Retrieval System that answers business-related questions using multiple data sources.

Unlike typical GenAI systems, this implementation uses pure Python (deterministic logic) for:

Data retrieval
Analysis
Business reasoning

This ensures:

✅ Zero hallucination
✅ Fully explainable outputs
✅ High accuracy
🎯 Objective

The system is designed to:

Retrieve data from multiple sources
Dynamically select sources based on query
Perform deterministic analysis
Generate business insights with citations
🏗️ Project Structure
project/
│── agent.py
│── data/
│   ├── sales.csv
│   ├── payroll.csv
│   └── report.txt
│── README.md
📂 Data Sources
📊 Structured Data
sales.csv → revenue, regions, trends
payroll.csv → salaries, expenses, headcount
📄 Unstructured Data
report.txt → explanations, root causes, summaries
⚙️ How It Works
1. User Input
python agent.py
2. Agent Decision Logic

The system determines which data sources to use:

[Agent Decision]
- Using sales.csv
- Using report.txt
3. Deterministic Reasoning

Python performs:

Trend analysis
Comparisons
Root cause extraction
Business insight generation

No LLM is used — all logic is rule-based.

4. Output with Citations

Example:

--- ANSWER ---
Revenue decreased from ₹136,750 to ₹112,210 (−17.9%), indicating declining performance.

--- SOURCES ---
Source: sales.csv (rows 2–25)
🧠 Agentic Decision Logic
Query Type	Source Used
Trend	sales.csv / payroll.csv
Comparison	sales.csv + payroll.csv
Why	report.txt + CSV
Summary	report.txt
Financial Health	Combined
🚀 Features
Multi-source retrieval (CSV + text)
Agent-based source selection
Deterministic reasoning engine
Business-level insights
CLI interface
Accurate citations
No LLM dependency
💡 Example Questions
How has revenue changed over the last quarter?
Compare salary expenses with revenue
Why did revenue decrease?
What are the key financial risks?
Is the business financially sustainable?
🛠️ Installation & Setup
Clone Repository
git clone <your-repo-link>
cd project
Install Dependencies
pip install pandas
Run the Project
python agent.py
📈 Example Output
Enter your question: Why did revenue decrease?

[Agent Decision]
- Using sales.csv
- Using report.txt

--- ANSWER ---
Revenue decreased from ₹136,750 to ₹112,210 due to competitive pressure and supply chain disruptions.

--- SOURCES ---
Source: sales.csv (rows 2–25)
Source: report.txt (paragraph 1)
🧪 Evaluation Criteria (Satisfied)
✔ Correct source selection
✔ Multi-source usage
✔ Accurate reasoning
✔ Clear citations
✔ Simple and structured implementation
🧠 Design Philosophy
Accuracy over automation
Explainability over black-box models
Deterministic logic over LLM dependency
🚀 Future Improvements
Add more datasets (inventory, customers)
Improve anomaly detection
Add visualization dashboards
Extend to API-based system
👩‍💻 Author

Swathiga S
AI Developer | Data Science Enthusiast
