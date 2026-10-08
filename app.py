import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="College Event Analytics",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset and Model
# --------------------------------------------------

df = pd.read_csv("data/college_event_ml_dataset.csv")

# Data cleaning
df["Distance_KM"] = df["Distance_KM"].fillna(
    df["Distance_KM"].median()
)

df["Technical_Interest"] = df["Technical_Interest"].fillna(
    df["Technical_Interest"].mode()[0]
)

model = joblib.load("models/attendance_model.pkl")

events_df = pd.read_csv("data/events.csv")

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("🎓 College Event Analytics")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "Dashboard",
        "Data Analytics",
        "Attendance Prediction",
        "Model Performance",
        "Project Management"
    ]
)
# --------------------------------------------------
# Title
# --------------------------------------------------

if page == "Dashboard":

    st.title("🎓 College Event Management & Attendance Analytics")

    st.write(
    "A Python-based system for analyzing college events, "
    "student participation, attendance and event performance."
    )

    st.divider()

# --------------------------------------------------
# Key Statistics
# --------------------------------------------------

    total_students = df["Student_ID"].nunique()
    total_events = df["Event_ID"].nunique()
    total_registrations = df["Registered"].sum()
    total_attendance = df["Attended"].sum()

    attendance_percentage = (
    total_attendance / total_registrations * 100
    if total_registrations > 0
    else 0
    )

    average_feedback = df["Average_Feedback"].mean()

# --------------------------------------------------
# Dashboard Cards
# --------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
     st.metric(
        "Students",
        total_students
    )

    with col2:
     st.metric(
        "Events",
        total_events
    )

    with col3:
     st.metric(
        "Registrations",
        int(total_registrations)
    )

    with col4:
     st.metric(
        "Attendance",
        f"{attendance_percentage:.2f}%"
    )

    with col5:
     st.metric(
        "Avg Feedback",
        f"{average_feedback:.2f}/5"
    )

    st.divider()

# --------------------------------------------------
# Event Type Distribution
# --------------------------------------------------

    st.subheader("📊 Event Type Distribution")

    event_counts = df["Event_Type"].value_counts()

    st.bar_chart(event_counts)

# --------------------------------------------------
# Department Attendance
# --------------------------------------------------

    st.subheader("📈 Attendance by Department")

    department_attendance = (
        df.groupby("Department")["Attended"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(department_attendance)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "This section shows how well the machine learning model "
        "performs in predicting student attendance."
    )

    st.divider()

    # Features used by the model

    y = df["Attended"]

    X = df[
        [
            "Previous_Events_Attended",
            "Previous_Attendance_Rate",
            "Technical_Interest",
            "Registered",
            "Notification_Received",
            "Distance_KM",
            "Registration_Channel",
            "Event_Type",
            "Department",
            "Year"
        ]
    ]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Prediction on test data

    y_pred = model.predict(X_test)

    # Performance metrics

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    # Explain metrics

    st.subheader("📖 Understanding the Metrics")

    st.write(
        "**Accuracy:** Percentage of total predictions that were correct."
    )

    st.write(
        "**Precision:** Among students predicted to attend, "
        "how many actually attended."
    )

    st.write(
        "**Recall:** Among students who actually attended, "
        "how many were correctly identified."
    )

    st.write(
        "**F1 Score:** A combined measure of precision and recall."
    )

    st.divider()

    # Confusion Matrix

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Not Attend", "Actual Attend"],
            columns=["Predicted Not Attend", "Predicted Attend"]
        )
    )

# --------------------------------------------------
# Attendance Prediction Page
# --------------------------------------------------

elif page == "Data Analytics":

    st.title("📊 Data Analytics")

    st.write(
        "Explore student participation, event attendance "
        "and feedback using interactive data analysis."
    )

    st.divider()

    st.subheader("🔎 Filters")

    col1, col2 = st.columns(2)

    with col1:

        selected_department = st.selectbox(
            "Select Department",
            ["All"] + sorted(df["Department"].unique().tolist())
        )

    with col2:

        selected_event = st.selectbox(
            "Select Event Type",
            ["All"] + sorted(df["Event_Type"].unique().tolist())
        )
            
        filtered_df = df.copy()

    if selected_department != "All":
        filtered_df = filtered_df[
            filtered_df["Department"] == selected_department
        ]

    if selected_event != "All":
        filtered_df = filtered_df[
            filtered_df["Event_Type"] == selected_event
        ]

        st.divider()

    st.subheader("📌 Selected Data Summary")

    total_records = len(filtered_df)

    attendance_rate = (
        filtered_df["Attended"].mean() * 100
        if total_records > 0
        else 0
    )

    avg_feedback = (
        filtered_df["Average_Feedback"].mean()
        if total_records > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Records",
            total_records
        )

    with col2:
        st.metric(
            "Attendance Rate",
            f"{attendance_rate:.2f}%"
        )

    with col3:
        st.metric(
            "Average Feedback",
            f"{avg_feedback:.2f}/5"
        )

        st.divider()

    st.subheader("📈 Attendance by Event")

    event_attendance = (
        filtered_df
        .groupby("Event_Name")["Attended"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    st.bar_chart(event_attendance)

    st.subheader("🏫 Attendance by Department")

    department_attendance = (
        filtered_df
        .groupby("Department")["Attended"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    st.bar_chart(department_attendance)

    st.subheader("⭐ Average Feedback by Event Type")

    feedback_analysis = (
        filtered_df
        .groupby("Event_Type")["Average_Feedback"]
        .mean()
        .round(2)
    )

    st.bar_chart(feedback_analysis)

    st.subheader("📊 Previous Attendance vs Current Attendance")

    attendance_comparison = (
        filtered_df
        .groupby("Attended")["Previous_Attendance_Rate"]
        .mean()
        .round(2)
    )

    attendance_comparison.index = [
        "Did Not Attend" if x == 0 else "Attended"
        for x in attendance_comparison.index
    ]

    st.bar_chart(attendance_comparison)

    st.subheader("📋 Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

elif page == "Attendance Prediction":

    st.title("🔮 Attendance Prediction")
    st.write(
        "Predict whether a student is likely to attend a college event "
        "using the trained machine learning model."
    )

    st.divider()

    # -----------------------------
    # Student Information
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique().tolist())
        )

        year = st.selectbox(
            "Academic Year",
            sorted(df["Year"].unique().tolist())
        )

        previous_events = st.number_input(
            "Previous Events Attended",
            min_value=0,
            max_value=50,
            value=5
        )

        previous_attendance = st.slider(
            "Previous Attendance Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        technical_interest = st.selectbox(
          "Technical Interest",
           df["Technical_Interest"].dropna().unique().tolist()
        )

    with col2:

        distance = st.number_input(
            "Distance from Venue (KM)",
            min_value=0.0,
            max_value=100.0,
            value=5.0
        )

        event_type = st.selectbox(
            "Event Type",
            sorted(df["Event_Type"].unique().tolist())
        )

        notification = st.selectbox(
            "Notification Received?",
            ["Yes", "No"]
        )

        registered = st.selectbox(
            "Student Registered for Event?",
            ["Yes", "No"]
        )

    st.divider()

    # -----------------------------
    # Prediction Button
    # -----------------------------

    if st.button("🔮 Predict Attendance"):

        # Convert Yes/No into 1/0
        registered_value = 1 if registered == "Yes" else 0
        notification_value = 1 if notification == "Yes" else 0

        # Create input dataframe
        input_data = pd.DataFrame({
            "Previous_Events_Attended": [previous_events],
            "Previous_Attendance_Rate": [previous_attendance],
            "Technical_Interest": [technical_interest],
            "Registered": [registered_value],
            "Notification_Received": [notification_value],
            "Distance_KM": [distance],
            "Registration_Channel": ["College Portal"],
            "Event_Type": [event_type],
            "Department": [department],
            "Year": [year]
        })

        # Make prediction
        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success(
                f"✅ Student is likely to ATTEND the event."
            )
        else:
            st.warning(
                f"❌ Student is likely to NOT ATTEND the event."
            )

        st.info(
            f"Attendance Probability: {probability * 100:.2f}%"
        )

elif page == "Project Management":

    st.title("📋 Project Management Dashboard")

    st.write(
        "Monitor event cost, attendance and schedule performance "
        "using project management metrics."
    )

    st.divider()

    # =========================
    # Overall Project Metrics
    # =========================

    total_events = events_df["Event_ID"].nunique()

    planned_budget = events_df["Planned_Budget"].sum()
    actual_cost = events_df["Actual_Cost"].sum()

    cost_variance = planned_budget - actual_cost

    planned_attendance = events_df["Planned_Attendance"].sum()
    actual_attendance = events_df["Actual_Attendance"].sum()

    attendance_efficiency = (
        actual_attendance / planned_attendance * 100
        if planned_attendance > 0
        else 0
    )

    planned_duration = events_df["Planned_Duration_Days"].sum()
    actual_duration = events_df["Actual_Duration_Days"].sum()

    schedule_variance = planned_duration - actual_duration

    schedule_efficiency = (
        planned_duration / actual_duration * 100
        if actual_duration > 0
        else 0
    )

    # =========================
    # KPI Cards
    # =========================

    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Events",
            total_events
        )

    with col2:
        st.metric(
            "Planned Budget",
            f"₹{planned_budget:,.0f}"
        )

    with col3:
        st.metric(
            "Actual Cost",
            f"₹{actual_cost:,.0f}"
        )

    with col4:
        st.metric(
            "Cost Variance",
            f"₹{cost_variance:,.0f}"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Planned Attendance",
            f"{planned_attendance:,}"
        )

    with col2:
        st.metric(
            "Actual Attendance",
            f"{actual_attendance:,}"
        )

    with col3:
        st.metric(
            "Attendance Efficiency",
            f"{attendance_efficiency:.2f}%"
        )

    with col4:
        st.metric(
            "Schedule Efficiency",
            f"{schedule_efficiency:.2f}%"
        )

    st.divider()

    # =========================
    # Budget Analysis
    # =========================

    st.subheader("💰 Budget Analysis")

    budget_data = events_df[
        ["Event_Name", "Planned_Budget", "Actual_Cost"]
    ].copy()

    budget_data = budget_data.set_index("Event_Name")

    st.bar_chart(budget_data)

    st.write(
        "**Cost Variance = Planned Budget − Actual Cost**"
    )

    if cost_variance >= 0:
        st.success(
            f"Overall project is within budget by ₹{cost_variance:,.0f}."
        )
    else:
        st.error(
            f"Overall project is over budget by ₹{abs(cost_variance):,.0f}."
        )

    st.divider()

    # =========================
    # Attendance Analysis
    # =========================

    st.subheader("👥 Planned vs Actual Attendance")

    attendance_data = events_df[
        ["Event_Name", "Planned_Attendance", "Actual_Attendance"]
    ].copy()

    attendance_data = attendance_data.set_index("Event_Name")

    st.bar_chart(attendance_data)

    st.write(
        f"Overall Attendance Efficiency: "
        f"**{attendance_efficiency:.2f}%**"
    )

    st.divider()

    # =========================
    # Schedule Analysis
    # =========================

    st.subheader("⏱️ Schedule Analysis")

    schedule_data = events_df[
        [
            "Event_Name",
            "Planned_Duration_Days",
            "Actual_Duration_Days"
        ]
    ].copy()

    schedule_data = schedule_data.set_index("Event_Name")

    st.bar_chart(schedule_data)

    st.write(
        f"Schedule Variance: "
        f"**{schedule_variance} days**"
    )

    if schedule_variance >= 0:
        st.success(
            f"Events were completed {schedule_variance} day(s) "
            "faster than planned overall."
        )
    else:
        st.warning(
            f"Events took {abs(schedule_variance)} more day(s) "
            "than planned overall."
        )

    st.divider()

    # =========================
    # Event Performance Table
    # =========================

    st.subheader("📋 Event Performance")

    performance_df = events_df[
        [
            "Event_ID",
            "Event_Name",
            "Planned_Budget",
            "Actual_Cost",
            "Planned_Attendance",
            "Actual_Attendance",
            "Planned_Duration_Days",
            "Actual_Duration_Days",
            "Average_Feedback"
        ]
    ].copy()

    st.dataframe(
        performance_df,
        use_container_width=True
    )

        # =========================
    # Risk Management
    # =========================

    st.divider()

    st.subheader("⚠️ Risk Management")

    st.write(
        "Identify, evaluate and monitor risks that may affect "
        "the successful completion of college events and the project."
    )

    risk_data = pd.DataFrame({
        "Risk ID": [
            "R01",
            "R02",
            "R03",
            "R04",
            "R05",
            "R06"
        ],

        "Risk": [
            "Technical Failure",
            "Low Student Participation",
            "Budget Overrun",
            "Data Loss",
            "Project Delay",
            "Prediction Model Error"
        ],

        "Probability": [
            3,
            3,
            2,
            2,
            3,
            3
        ],

        "Impact": [
            5,
            4,
            4,
            5,
            4,
            3
        ],

        "Mitigation": [
            "Keep backup devices and internet connection",
            "Send reminders and promote events",
            "Monitor expenses regularly",
            "Maintain regular data backups",
            "Use proper scheduling and time buffers",
            "Test and validate the ML model"
        ],

        "Status": [
            "Monitoring",
            "Monitoring",
            "Monitoring",
            "Mitigated",
            "Monitoring",
            "Monitoring"
        ]
    })

    # Calculate risk score

    risk_data["Risk Score"] = (
        risk_data["Probability"] *
        risk_data["Impact"]
    )

    # Determine risk level

    def get_risk_level(score):

        if score >= 15:
            return "High"

        elif score >= 8:
            return "Medium"

        else:
            return "Low"

    risk_data["Risk Level"] = (
        risk_data["Risk Score"]
        .apply(get_risk_level)
    )

    # Display risk table

    st.dataframe(
        risk_data,
        use_container_width=True
    )

    st.divider()

    # Risk summary

    high_risks = len(
        risk_data[risk_data["Risk Level"] == "High"]
    )

    medium_risks = len(
        risk_data[risk_data["Risk Level"] == "Medium"]
    )

    low_risks = len(
        risk_data[risk_data["Risk Level"] == "Low"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "High Risks",
            high_risks
        )

    with col2:
        st.metric(
            "Medium Risks",
            medium_risks
        )

    with col3:
        st.metric(
            "Low Risks",
            low_risks
        )

    # Risk score chart

    st.subheader("📊 Risk Score Analysis")

    risk_chart = risk_data[
        ["Risk", "Risk Score"]
    ].set_index("Risk")

    st.bar_chart(risk_chart)

    st.info(
        "Risk Score = Probability × Impact"
    )

        # =========================
    # Resource Management
    # =========================

    st.divider()

    st.subheader("👨‍💻 Resource Management")

    st.write(
        "Plan and monitor the human, technical and physical resources "
        "required to conduct college events and complete the project."
    )

    resource_data = pd.DataFrame({
        "Resource": [
            "Project Manager",
            "Event Coordinator",
            "Technical Team",
            "Registration Team",
            "Faculty Coordinators",
            "Volunteers",
            "Venue",
            "Computers",
            "Audio/Visual Equipment",
            "Internet Connection"
        ],

        "Resource Type": [
            "Human",
            "Human",
            "Human",
            "Human",
            "Human",
            "Human",
            "Facility",
            "Equipment",
            "Equipment",
            "Technical"
        ],

        "Required": [
            1,
            4,
            5,
            4,
            6,
            15,
            2,
            20,
            5,
            2
        ],

        "Available": [
            1,
            4,
            5,
            4,
            6,
            18,
            2,
            20,
            5,
            2
        ]
    })

    # Calculate utilization

    resource_data["Utilization (%)"] = (
        resource_data["Required"] /
        resource_data["Available"] * 100
    ).round(2)

    # Determine resource status

    def resource_status(row):

        if row["Available"] >= row["Required"]:
            return "Available"

        else:
            return "Shortage"

    resource_data["Status"] = resource_data.apply(
        resource_status,
        axis=1
    )

    st.dataframe(
        resource_data,
        use_container_width=True
    )

    st.divider()

    # Resource summary

    total_resources = len(resource_data)

    available_resources = len(
        resource_data[
            resource_data["Status"] == "Available"
        ]
    )

    shortage_resources = len(
        resource_data[
            resource_data["Status"] == "Shortage"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Resource Categories",
            total_resources
        )

    with col2:
        st.metric(
            "Available",
            available_resources
        )

    with col3:
        st.metric(
            "Shortage",
            shortage_resources
        )

    st.subheader("📊 Resource Requirement")

    resource_chart = resource_data[
        ["Resource", "Required", "Available"]
    ].set_index("Resource")

    st.bar_chart(resource_chart)

        # =========================
    # Project Planning
    # =========================

    st.divider()

    st.subheader("📝 Project Planning")

    st.write(
        "Major activities required to develop and operate the "
        "College Event Management and Attendance Analytics System."
    )

    planning_data = pd.DataFrame({
        "Phase": [
            "Requirement Analysis",
            "Data Collection",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Machine Learning",
            "Website Development",
            "Testing",
            "Deployment"
        ],

        "Duration (Days)": [
            3,
            5,
            4,
            4,
            6,
            8,
            4,
            2
        ],

        "Responsible Team": [
            "Project Manager",
            "Data Team",
            "Data Team",
            "Data Team",
            "ML Team",
            "Development Team",
            "Testing Team",
            "Deployment Team"
        ],

        "Status": [
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "In Progress",
            "In Progress",
            "Planned"
        ]
    })

    st.dataframe(
        planning_data,
        use_container_width=True
    )

    st.subheader("📅 Project Phase Duration")

    phase_chart = planning_data[
        ["Phase", "Duration (Days)"]
    ].set_index("Phase")

    st.bar_chart(phase_chart)

        # =========================
    # Project Performance Summary
    # =========================

    st.divider()

    st.subheader("📌 Project Performance Summary")

    st.write(
        "Overall assessment of the project based on cost, attendance, "
        "schedule, resources and risk."
    )

    # -------------------------
    # Determine project status
    # -------------------------

    # Budget status
    if cost_variance >= 0:
        budget_status = "On Budget"
    else:
        budget_status = "Over Budget"

    # Attendance status
    if attendance_efficiency >= 100:
        attendance_status = "Above Target"
    elif attendance_efficiency >= 90:
        attendance_status = "Near Target"
    else:
        attendance_status = "Below Target"

    # Schedule status
    if schedule_variance >= 0:
        schedule_status = "On Schedule"
    else:
        schedule_status = "Delayed"

    # Resource status
    if shortage_resources == 0:
        resource_status = "Sufficient"
    else:
        resource_status = "Shortage"

    # Risk status
    if high_risks == 0:
        risk_status = "Low Risk"
    elif high_risks <= 2:
        risk_status = "Moderate Risk"
    else:
        risk_status = "High Risk"

    # -------------------------
    # Display project status
    # -------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Budget",
            budget_status
        )

    with col2:
        st.metric(
            "Attendance",
            attendance_status
        )

    with col3:
        st.metric(
            "Schedule",
            schedule_status
        )

    with col4:
        st.metric(
            "Resources",
            resource_status
        )

    with col5:
        st.metric(
            "Risk",
            risk_status
        )

    st.divider()

    # -------------------------
    # Performance table
    # -------------------------

    summary_data = pd.DataFrame({
        "Project Area": [
            "Cost",
            "Attendance",
            "Schedule",
            "Resources",
            "Risk"
        ],

        "Performance": [
            budget_status,
            attendance_status,
            schedule_status,
            resource_status,
            risk_status
        ],

        "Key Result": [
            f"Variance: ₹{cost_variance:,.0f}",
            f"Efficiency: {attendance_efficiency:.2f}%",
            f"Variance: {schedule_variance} days",
            f"{available_resources}/{total_resources} sufficient",
            f"{high_risks} high-risk item(s)"
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True
    )

        # =========================
    # Gantt Chart
    # =========================

    st.divider()

    st.subheader("📅 Project Gantt Chart")

    st.write(
        "The Gantt chart shows the planned sequence and duration "
        "of major project activities."
    )

    # Create start and end days

    gantt_data = planning_data.copy()

    gantt_data["Start Day"] = (
        gantt_data["Duration (Days)"].cumsum()
        - gantt_data["Duration (Days)"]
        + 1
    )

    gantt_data["End Day"] = (
        gantt_data["Start Day"]
        + gantt_data["Duration (Days)"]
        - 1
    )

    # Create chart

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, row in gantt_data.iterrows():

        ax.barh(
            row["Phase"],
            row["Duration (Days)"],
            left=row["Start Day"]
        )

        ax.text(
            row["Start Day"] + row["Duration (Days)"] / 2,
            i,
            f'{row["Duration (Days)"]} days',
            ha="center",
            va="center"
        )

    ax.set_xlabel("Project Day")
    ax.set_ylabel("Project Phase")
    ax.set_title("College Event Analytics Project Schedule")

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("📋 Gantt Schedule Details")

    st.dataframe(
        gantt_data[
            [
                "Phase",
                "Start Day",
                "End Day",
                "Duration (Days)",
                "Status"
            ]
        ],
        use_container_width=True
    )

        # =========================
    # PERT Analysis
    # =========================

    st.divider()

    st.subheader("📐 PERT Analysis")

    st.write(
        "PERT (Program Evaluation and Review Technique) is used to "
        "estimate project activity durations under uncertainty."
    )

    pert_data = pd.DataFrame({
        "Activity": [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H"
        ],

        "Activity Name": [
            "Requirement Analysis",
            "Data Collection",
            "Data Cleaning",
            "EDA",
            "Machine Learning",
            "Website Development",
            "Testing",
            "Deployment"
        ],

        "Predecessor": [
            "-",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ],

        "Optimistic (O)": [
            2,
            3,
            2,
            2,
            4,
            6,
            2,
            1
        ],

        "Most Likely (M)": [
            3,
            5,
            4,
            4,
            6,
            8,
            4,
            2
        ],

        "Pessimistic (P)": [
            5,
            8,
            7,
            7,
            10,
            12,
            7,
            4
        ]
    })

    # Calculate expected time

    pert_data["Expected Time"] = (
        (
            pert_data["Optimistic (O)"]
            + 4 * pert_data["Most Likely (M)"]
            + pert_data["Pessimistic (P)"]
        ) / 6
    ).round(2)

    st.dataframe(
        pert_data,
        use_container_width=True
    )

    st.divider()

    # PERT expected project duration

    expected_project_duration = pert_data[
        "Expected Time"
    ].sum()

    st.metric(
        "Expected Project Duration",
        f"{expected_project_duration:.2f} days"
    )

    st.info(
        "PERT Expected Time = (Optimistic + 4 × Most Likely + Pessimistic) / 6"
    )

        # =========================
    # CPM Analysis
    # =========================

    st.divider()

    st.subheader("🔴 CPM - Critical Path Method")

    st.write(
        "CPM identifies the sequence of activities that determines "
        "the minimum time required to complete the project."
    )

    critical_path = [
        "A - Requirement Analysis",
        "B - Data Collection",
        "C - Data Cleaning",
        "D - EDA",
        "E - Machine Learning",
        "F - Website Development",
        "G - Testing",
        "H - Deployment"
    ]

    st.write("### Critical Path")

    st.success(
        " → ".join(critical_path)
    )

    st.metric(
        "Critical Path Duration",
        f"{expected_project_duration:.2f} days"
    )

    st.warning(
        "Activities on the critical path have no planned time buffer "
        "in this simplified project network. A delay in one of these "
        "activities can delay the overall project."
    )