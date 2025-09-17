import streamlit as st
import pandas as pd
import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- Prompts Dictionary ---
PROMPTS = {
    "dashboard_narration": {
        "executive_summary": "Summarize the key performance indicators (KPIs) and any notable anomalies from the following dashboard data into a concise executive-level briefing: {dashboard_data}",
        "stakeholder_summary": "Convert the provided performance dashboard into a 5-bullet point summary for stakeholders, using plain English. Focus on the most critical trends and insights: {dashboard_data}",
        "churn_retention_summary": "Generate a dashboard narration explaining the churn rate, retention trends, and customer segment performance based on this data: {dashboard_data}",
        "c_suite_vs_managers_summary": "Create two distinct summaries of the dashboard. The first should be a high-level strategic overview for the C-suite. The second should be a more detailed, operational summary for managers: {dashboard_data}",
        "meeting_script": "Write a script for a Business Analyst to present the findings from this dashboard in a monthly review meeting. The script should be clear, concise, and engaging: {dashboard_data}"
    },
    "trend_interpretation": {
        "customer_acquisition_patterns": "Interpret this dataset to identify patterns in customer acquisition. What are the potential business implications of these patterns? Dataset: {dataset}",
        "quarterly_performance_summary": "Using the quarterly data on product usage, generate a summary that highlights performance trends and identifies key drop-off points. Data: {dataset}",
        "seasonal_trend_explanation": "Detect seasonal trends from this retail sales dataset and explain them in business terms. What factors might be driving these trends? Dataset: {dataset}",
        "hypothesis_generation": "Analyze the attached chart showing declining support ticket resolution times. Propose three distinct hypotheses that could explain this trend. Chart data: {chart_data}",
        "saas_kpi_interpretation": "Create a prompt to help an AI interpret key SaaS metrics such as ARPU (Average Revenue Per User), MRR (Monthly Recurring Revenue), and CAC (Customer Acquisition Cost) from a SaaS metrics dashboard. The prompt should guide the AI to explain what these metrics are, how they are performing, and what their business impact is. Dashboard data: {dashboard_data}"
    },
    "data_storytelling": {
        "revenue_narrative": "Generate a compelling narrative using the provided sales data to explain why Q3 outperformed Q2 in both revenue and conversions. Data: {sales_data}",
        "customer_success_story": "Turn the following metrics into a persuasive customer success story suitable for an investor pitch. The story should highlight growth, engagement, and customer satisfaction. Metrics: {metrics}",
        "operational_efficiency_story": "Write a 200-word story that explains the operational efficiency gains achieved through AI-powered automation, based on the following data. Data: {automation_data}",
        "visualization_recommendation": "Given the following dataset and its metrics, recommend the best chart type (e.g., bar chart, line graph, pie chart) for visualizing each metric. Explain your reasoning for each recommendation. Dataset: {dataset}",
        "digital_transformation_case_study": "Build a mini case study using the provided before-and-after analytics to illustrate the success of a digital transformation initiative. The case study should be concise and impact-oriented. Analytics data: {analytics_data}"
    },
    "tool_integration": {
        "excel_formula_explanation": "Generate an Excel formula explanation guide based on this sales spreadsheet. The guide should explain the formulas used for summary statistics and trend analysis. Spreadsheet data: {spreadsheet_data}",
        "sql_summary_generation": "Write a SQL query that summarizes customer orders by region and year, suitable for business intelligence reporting. The query should be well-commented and easy to understand. Schema/Data: {db_data}",
        "power_bi_report_explanation": "Generate an AI-powered explanation for a Power BI report that shows monthly cost variances across different departments. The explanation should be clear and highlight the key drivers of these variances. Report data: {report_data}",
        "tableau_dashboard_summary": "Convert this Tableau dashboard into a descriptive summary. Based on the summary, list three actionable business recommendations. Dashboard data: {dashboard_data}",
        "data_cleaning_prompt": "Create a prompt to help an AI clean and standardize tabular data imported from Excel into a BI dashboard. The prompt should guide the AI on how to handle missing values, correct data types, and resolve inconsistencies. Sample data: {sample_data}"
    },
    "brd_frd_srs_generation": {
        "brd_template": "Generate a comprehensive Business Requirements Document (BRD) template for a cloud-based document management system. The template should include sections for business objectives, scope (in and out), stakeholder list, functional and non-functional requirements, and success metrics.",
        "frd_template": "Create a structured Functional Requirements Document (FRD) template. The template should have placeholders for functional requirements, use cases, user roles and permissions, and system workflows.",
        "srs_template": "Draft a simple Software Requirements Specification (SRS) outline for a mobile application. The outline should include an introduction, system features, user interfaces, and data requirements.",
        "requirement_pack_generation": "Generate a multi-document requirement pack for a banking loan origination system. The pack should include a BRD, FRD, and SRS, all tailored to the specific needs of a financial institution.",
        "brd_from_notes_prompt": "Create a prompt that allows an AI to convert unstructured stakeholder notes into a pre-filled Business Requirements Document (BRD) template. The prompt should guide the AI to identify key information and map it to the correct sections of the BRD. Notes: {notes}"
    },
    "section_wise_prompts": {
        "business_objectives_section": "Generate a 'Business Objectives' section for a BRD focused on implementing a real-time customer support chatbot. The objectives should be SMART (Specific, Measurable, Achievable, Relevant, Time-bound).",
        "scope_section": "Write the 'Scope' and 'Out of Scope' sections for a project aimed at enhancing the UI/UX of an eCommerce platform. Be specific about the features and pages that will be included and excluded.",
        "functional_requirements_section": "Draft the 'Functional Requirements' section of an FRD for a warehouse inventory tracking system. The requirements should be detailed and cover all key functionalities like stock entry, tracking, and reporting.",
        "system_features_section": "Create the 'System Features' section of an SRS for an employee onboarding application. List and describe the main features, such as user registration, document submission, and training module access.",
        "assumptions_and_constraints_section": "Generate a list of 'Assumptions and Constraints' for inclusion in a BRD about automating payroll processing. This should cover technical, business, and operational aspects."
    },
    "gap_analysis": {
        "brd_charter_gap": "Identify requirement gaps between the current Business Requirements Document (BRD) and the project charter for a logistics management project. Highlight any misalignments or missing information. BRD: {brd}, Charter: {charter}",
        "frs_integration_gap": "Analyze the given Functional Requirements Specification (FRS) and highlight any missing system integration details. What other systems should be considered for integration? FRS: {frs}",
        "brd_gap_checklist": "Create a checklist of typical BRD gaps that can delay development or quality assurance (QA) sign-off. The checklist should be organized by section (e.g., Scope, Requirements, Stakeholders).",
        "srs_stakeholder_gap": "Generate a side-by-side comparison showing coverage gaps between the Software Requirements Specification (SRS) and the stakeholder expectations. Where do the requirements fall short of what the stakeholders want? SRS: {srs}, Expectations: {expectations}",
        "stakeholder_briefing_gaps": "Write a stakeholder-ready briefing that highlights the three key requirement gaps discovered during a peer review. The briefing should be concise, clear, and propose next steps for addressing the gaps. Gaps: {gaps}"
    },
    "summary_writing": {
        "brd_one_pager": "Summarize a 15-page Business Requirements Document (BRD) into an executive-friendly one-pager. The summary should cover key goals, risks, and timelines. BRD: {brd}",
        "frd_conclusion": "Generate a conclusion section for a Functional Requirements Document (FRD). The conclusion should summarize the project's scope, key features, and any notable risks or dependencies. FRD: {frd}",
        "srs_project_overview": "Create a project overview summary paragraph for the Software Requirements Specification (SRS) of a mobile fitness tracking application. The overview should be suitable for the introduction section.",
        "brd_summary_non_technical": "Generate a BRD summary that explains the system's goals and benefits to non-technical stakeholders. Avoid jargon and focus on the business value.",
        "brd_appendix": "Draft a structured appendix for a Business Requirements Document (BRD). The appendix should include a glossary of terms, a list of acronyms, and references to related documents."
    }
}


# --- Agent & Backend Logic ---

def get_llm(api_key):
    """Initializes and returns the AzureChatOpenAI instance."""
    # In a production environment, it's highly recommended to use environment variables
    # for endpoint and version, and st.secrets for the API key.
    if not api_key:
        st.error("Azure OpenAI API Key is required.")
        st.stop()
        
    try:
        # User is expected to have AZURE_OPENAI_ENDPOINT and OPENAI_API_VERSION set as environment variables.
        # This is a common practice for Azure SDKs.
        llm = AzureChatOpenAI(
            openai_api_version=os.environ.get("OPENAI_API_VERSION", "2023-05-15"), # Default to a common version
            azure_deployment="gpt-4o", # As requested by the user
            openai_api_key=api_key,
            temperature=0.7,
            max_tokens=4000,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize the language model: {e}")
        st.stop()

def analyze_data(api_key, data, brd_content, prompt_category, prompt_key, user_question):
    """
    This function houses the agent logic to perform data analysis.
    """
    llm = get_llm(api_key)

    # 1. Select the base prompt from the dictionary
    prompt_template_str = PROMPTS.get(prompt_category, {}).get(prompt_key)
    if not prompt_template_str:
        return "Error: Invalid prompt selection."

    # 2. Prepare the data and context
    # For large datasets, providing a summary or head is better than the full CSV.
    data_head = data.head(20).to_string()
    data_summary = str(data.describe(include='all'))

    # 3. Construct a comprehensive final prompt
    final_prompt_template = f'''
You are an expert data analyst and business consultant. Your task is to provide a detailed analysis based on the user's request.

**User's Goal/Question:**
{{user_question}}

**Selected Task:**
"{prompt_key}" - This is the high-level task you should focus on.

**Core Prompt Template:**
"{prompt_template_str}"

**Business Context (from BRD/document, if provided):**
{{brd_context}}

**Dataset Summary:**
Here is a statistical summary of the provided dataset:
{{data_summary}}

**Dataset Preview (first 20 rows):**
{{data_head}}

---
**Your Response:**
Please generate a comprehensive, well-structured response that directly addresses the user's question and fulfills the task's requirements.
Use the provided data and business context to formulate your analysis.
Be clear, insightful, and provide actionable recommendations where applicable.
'''
    
    # 4. Create and invoke the LangChain chain
    try:
        prompt = ChatPromptTemplate.from_template(final_prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "user_question": user_question or "Please perform the selected analysis.",
            "prompt_key": prompt_key,
            "prompt_template_str": prompt_template_str,
            "brd_context": brd_content or "No additional business context was provided.",
            "data_summary": data_summary,
            "data_head": data_head
        })
        
        return response
    except Exception as e:
        return f"An error occurred while communicating with the AI model: {e}"


# --- Streamlit UI ---

st.set_page_config(layout="wide")

st.title("🤖 Intelligent Multi-Agent System")
st.markdown("This application uses AI agents to perform data analysis, generate documentation, and provide actionable insights.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Inputs")
    
    # 1. API Key (optional but good practice)
    # In a real app, use st.secrets for this.
    openai_api_key = st.text_input("Enter your Azure OpenAI API Key", type="password", help="Your key is not stored.")

    # 2. Dataset Upload
    st.subheader("1. Upload Dataset")
    uploaded_dataset = st.file_uploader(
        "Upload your data file (CSV or Excel)", 
        type=["csv", "xlsx"]
    )

    # 3. BRD/Document Upload
    st.subheader("2. Upload Business Document (Optional)")
    uploaded_document = st.file_uploader(
        "Upload a BRD, problem statement, or other relevant text file", 
        type=["txt", "md"]
    )

    # 4. Analysis Selection
    st.subheader("3. Select Analysis Type")
    analysis_category = st.selectbox(
        "Choose a category",
        options=list(PROMPTS.keys())
    )
    
    if analysis_category:
        analysis_task = st.selectbox(
            "Choose a specific task",
            options=list(PROMPTS[analysis_category].keys())
        )

    # 5. User Prompt
    st.subheader("4. Ask a Question")
    user_prompt = st.text_area(
        "Enter your specific question or prompt here.",
        height=100
    )
    
    # 6. Analyze Button
    analyze_button = st.button("🚀 Analyze Now")


# --- Main Content Area ---

if analyze_button:
    if not openai_api_key:
        st.warning("Please enter your Azure OpenAI API Key in the sidebar.")
    elif uploaded_dataset is None:
        st.warning("Please upload a dataset to begin the analysis.")
    else:
        with st.spinner("Analyzing your data... Please wait."):
            try:
                # Load the dataset into a pandas DataFrame
                if uploaded_dataset.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_dataset)
                else:
                    df = pd.read_excel(uploaded_dataset)
                
                st.subheader("Uploaded Dataset Preview")
                st.dataframe(df.head())

                # Read the content of the uploaded document
                brd_text = ""
                if uploaded_document is not None:
                    brd_text = uploaded_document.read().decode("utf-8")
                    with st.expander("View Uploaded Document Content"):
                        st.text(brd_text[:500] + "...") # Preview first 500 chars

                # --- Trigger Analysis ---
                generated_text = analyze_data(
                    api_key=openai_api_key,
                    data=df,
                    brd_content=brd_text,
                    prompt_category=analysis_category,
                    prompt_key=analysis_task,
                    user_question=user_prompt
                )

                st.subheader("📊 Generated Analysis & Insights")
                st.markdown(generated_text)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

else:
    st.info("Please upload a dataset and select an analysis type from the sidebar to get started.")
