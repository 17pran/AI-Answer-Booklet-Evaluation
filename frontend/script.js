/* =====================================================
   AI ANSWER SCRIPT EVALUATION
   FRONTEND-ONLY VERSION

   No database.
   No backend connection.

   Dummy evaluation results are used for now.

   Later we will connect this file to FastAPI.
===================================================== */



/* =====================================================
   APPLICATION STATE
===================================================== */


/*
    These variables temporarily hold the uploaded
    reference files.

    They will remain available while the teacher
    evaluates multiple students.
*/

let answerKeyFile = null;

let textbookFile = null;



/*
    Information about the currently selected student.
*/

let currentStudent = {

    name: "",

    usn: "",

    answerScript: null

};



/*
    This stores the latest evaluation result.

    Later this will come from your AI backend.
*/

let currentEvaluation = null;



/*
    Chart object.

    We keep this so that we can destroy the old chart
    before creating a new one.
*/

let marksChart = null;



/* =====================================================
   PAGE NAVIGATION
===================================================== */


function showPage(pageId) {

    /*
        Hide every page.
    */

    const pages =
        document.querySelectorAll(".page");


    pages.forEach(function(page) {

        page.classList.remove("active");

    });


    /*
        Show requested page.
    */

    document
        .getElementById(pageId)
        .classList.add("active");

}



/* =====================================================
   LOGIN
===================================================== */


document
    .getElementById("loginForm")
    .addEventListener("submit", function(event) {


        /*
            Prevent normal form submission.
        */

        event.preventDefault();


        const username =
            document
                .getElementById("username")
                .value
                .trim();


        const password =
            document
                .getElementById("password")
                .value;


        /*
            FRONTEND-ONLY DEMO LOGIN

            Username:
            teacher

            Password:
            teacher123
        */


        if (
            username === "teacher" &&
            password === "teacher123"
        ) {

            showPage("referencePage");

        }

        else {

            alert(
                "Invalid username or password.\n\n" +
                "Demo login:\n" +
                "Username: teacher\n" +
                "Password: teacher123"
            );

        }

    });



/* =====================================================
   REFERENCE FILE UPLOAD
===================================================== */


function handleReferenceFile(
    inputId,
    displayId
) {


    const input =
        document.getElementById(inputId);


    const display =
        document.getElementById(displayId);


    /*
        Check whether a file was selected.
    */

    if (input.files.length === 0) {

        return;

    }


    const file =
        input.files[0];


    /*
        Store the file.
    */

    if (inputId === "answerKey") {

        answerKeyFile = file;

    }


    if (inputId === "textbook") {

        textbookFile = file;

    }


    /*
        Display filename.
    */

    display.textContent =
        "✓ " + file.name;


    /*
        Check whether both files
        are available.
    */

    updateReferenceStatus();

}



/* =====================================================
   REFERENCE STATUS
===================================================== */


function updateReferenceStatus() {


    const status =
        document.getElementById(
            "referenceStatus"
        );


    if (
        answerKeyFile !== null &&
        textbookFile !== null
    ) {

        status.classList.remove("hidden");

    }

    else {

        status.classList.add("hidden");

    }

}



/* =====================================================
   GO TO STUDENT PAGE
===================================================== */


function goToStudentPage() {


    /*
        Both reference files are required.
    */

    if (answerKeyFile === null) {

        alert(
            "Please upload the Answer Key PDF first."
        );

        return;

    }


    if (textbookFile === null) {

        alert(
            "Please upload the Textbook PDF first."
        );

        return;

    }


    /*
        Move to Student Details page.
    */

    showPage("studentPage");

}



/* =====================================================
   STUDENT ANSWER SCRIPT
===================================================== */


function handleStudentFile() {


    const input =
        document.getElementById(
            "answerScript"
        );


    const display =
        document.getElementById(
            "answerScriptName"
        );


    if (input.files.length === 0) {

        return;

    }


    const file =
        input.files[0];


    /*
        Store the answer script.
    */

    currentStudent.answerScript = file;


    /*
        Show filename.
    */

    display.textContent =
        "✓ " + file.name;

}



/* =====================================================
   EVALUATE STUDENT
===================================================== */


function evaluateStudent() {


    const name =
        document
            .getElementById("studentName")
            .value
            .trim();


    const usn =
        document
            .getElementById("studentUSN")
            .value
            .trim();


    const answerScript =
        document
            .getElementById("answerScript")
            .files[0];


    /*
        Validate student name.
    */

    if (name === "") {

        alert(
            "Please enter the student's name."
        );

        return;

    }


    /*
        Validate USN.
    */

    if (usn === "") {

        alert(
            "Please enter the student's USN."
        );

        return;

    }


    /*
        Validate answer script.
    */

    if (!answerScript) {

        alert(
            "Please upload the student's answer script PDF."
        );

        return;

    }


    /*
        Store current student.
    */

    currentStudent = {

        name: name,

        usn: usn,

        answerScript: answerScript

    };



    /* =================================================
       DUMMY AI RESULT
       
       This is temporary.

       Later your FastAPI backend will return
       these values.
    ================================================= */


    currentEvaluation = {

        obtained: 38,

        total: 50,


        questions: [

            {
                question: "Q1",

                max: 5,

                obtained: 4,

                evaluation: "Good answer"
            },


            {
                question: "Q2",

                max: 5,

                obtained: 5,

                evaluation: "Excellent answer"
            },


            {
                question: "Q3",

                max: 10,

                obtained: 7,

                evaluation: "Good understanding"
            },


            {
                question: "Q4",

                max: 10,

                obtained: 8,

                evaluation: "Very good answer"
            },


            {
                question: "Q5",

                max: 10,

                obtained: 6,

                evaluation: "Needs improvement"
            },


            {
                question: "Q6",

                max: 10,

                obtained: 8,

                evaluation: "Good answer"
            }

        ]

    };


    /*
        Display the result.
    */

    displayResult();

}



/* =====================================================
   DISPLAY RESULT
===================================================== */


function displayResult() {


    const result =
        currentEvaluation;


    /*
        Student information.
    */

    document
        .getElementById(
            "resultStudentName"
        )
        .textContent =
            currentStudent.name;


    document
        .getElementById(
            "resultUSN"
        )
        .textContent =
            "USN: " + currentStudent.usn;



    /*
        Marks.
    */

    document
        .getElementById(
            "obtainedMarks"
        )
        .textContent =
            result.obtained;


    document
        .getElementById(
            "totalMarks"
        )
        .textContent =
            result.total;



    /*
        Percentage.
    */

    const percentage =
        (
            result.obtained /
            result.total
        ) * 100;


    document
        .getElementById(
            "percentage"
        )
        .textContent =
            percentage.toFixed(1) + "%";



    /*
        Rating.
    */

    document
        .getElementById(
            "rating"
        )
        .textContent =
            getRating(percentage);



    /*
        Create chart.
    */

    createChart(
        result.obtained,
        result.total
    );



    /*
        Move to Page 4.
    */

    showPage("resultPage");

}



/* =====================================================
   RATING
===================================================== */


function getRating(percentage) {


    if (percentage >= 90) {

        return "Excellent Performance";

    }


    if (percentage >= 75) {

        return "Very Good Performance";

    }


    if (percentage >= 60) {

        return "Good Performance";

    }


    if (percentage >= 40) {

        return "Average Performance";

    }


    return "Needs Improvement";

}



/* =====================================================
   CREATE PIE / DOUGHNUT CHART
===================================================== */


function createChart(
    obtained,
    total
) {


    const remaining =
        total - obtained;


    const canvas =
        document.getElementById(
            "marksChart"
        );


    const context =
        canvas.getContext("2d");


    /*
        Destroy previous chart.

        This is important when the teacher
        evaluates the next student.
    */

    if (marksChart !== null) {

        marksChart.destroy();

    }


    /*
        Create new chart.
    */

    marksChart = new Chart(
        context,
        {

            type: "doughnut",


            data: {

                labels: [

                    "Marks Obtained",

                    "Marks Remaining"

                ],


                datasets: [

                    {

                        data: [

                            obtained,

                            remaining

                        ]

                    }

                ]

            },


            options: {

                responsive: true,

                maintainAspectRatio: false,

                cutout: "68%",


                plugins: {

                    legend: {

                        position: "bottom"

                    }

                }

            }

        }

    );

}



/* =====================================================
   DOWNLOAD DETAILS
===================================================== */


function downloadDetails() {


    /*
        Get current student.
    */

    const name =
        currentStudent.name;


    const usn =
        currentStudent.usn;



    /*
        Get marks.
    */

    const obtained =
        currentEvaluation.obtained;


    const total =
        currentEvaluation.total;


    const percentage =
        (
            obtained /
            total
        ) * 100;


    const rating =
        getRating(percentage);



    /*
        Start report.
    */

    let report = "";


    report +=
        "====================================================\n";

    report +=
        "           AI ANSWER SCRIPT EVALUATION\n";

    report +=
        "====================================================\n\n";


    report +=
        "STUDENT DETAILS\n";

    report +=
        "----------------------------------------------------\n";


    report +=
        "Student Name : " +
        name +
        "\n";


    report +=
        "USN          : " +
        usn +
        "\n\n";


    report +=
        "OVERALL RESULT\n";

    report +=
        "----------------------------------------------------\n";


    report +=
        "Marks        : " +
        obtained +
        " / " +
        total +
        "\n";


    report +=
        "Percentage    : " +
        percentage.toFixed(1) +
        "%\n";


    report +=
        "Rating        : " +
        rating +
        "\n\n";



    /*
        Question-wise details.

        These are NOT displayed on Page 4.

        They are included only in the downloaded report.
    */

    report +=
        "QUESTION-WISE EVALUATION\n";

    report +=
        "====================================================\n\n";


    currentEvaluation.questions.forEach(
        function(question) {


            report +=
                question.question +
                "\n";


            report +=
                "Maximum Marks : " +
                question.max +
                "\n";


            report +=
                "Obtained Marks: " +
                question.obtained +
                "\n";


            report +=
                "Evaluation    : " +
                question.evaluation +
                "\n";


            report +=
                "----------------------------------------------------\n";

        }
    );



    report +=
        "\nGenerated by AI Answer Evaluator\n";



    /*
        Convert report into downloadable file.
    */

    const blob =
        new Blob(
            [report],
            {
                type: "text/plain"
            }
        );


    const url =
        URL.createObjectURL(blob);


    const link =
        document.createElement("a");


    link.href = url;


    link.download =
        usn +
        "_evaluation_details.txt";


    document.body.appendChild(link);


    link.click();


    document.body.removeChild(link);


    URL.revokeObjectURL(url);

}



/* =====================================================
   NEXT STUDENT
===================================================== */


function nextStudent() {


    /*
        VERY IMPORTANT:

        We DO NOT clear:

        answerKeyFile
        textbookFile

        Therefore the teacher does NOT have to
        upload them again.
    */


    /*
        Clear only student information.
    */

    document
        .getElementById("studentName")
        .value = "";


    document
        .getElementById("studentUSN")
        .value = "";


    document
        .getElementById("answerScript")
        .value = "";


    document
        .getElementById("answerScriptName")
        .textContent =
            "No file selected";


    /*
        Reset current student.
    */

    currentStudent = {

        name: "",

        usn: "",

        answerScript: null

    };


    /*
        Reset evaluation.
    */

    currentEvaluation = null;


    /*
        Go directly to Student Details.

        NOT to Reference Materials.
    */

    showPage("studentPage");

}



/* =====================================================
   EXIT EVALUATION
===================================================== */


function exitEvaluation() {


    /*
        Current student data can be cleared.
    */

    currentStudent = {

        name: "",

        usn: "",

        answerScript: null

    };


    currentEvaluation = null;


    /*
        IMPORTANT:

        Answer key and textbook are NOT cleared.

        Teacher returns to Page 2.
    */

    showPage("referencePage");

}



/* =====================================================
   BACK TO REFERENCE MATERIALS
===================================================== */


function backToReference() {


    /*
        This is useful if teacher accidentally
        wants to return to Page 2.

        Reference files remain available.
    */

    showPage("referencePage");

}



/* =====================================================
   LOGOUT
===================================================== */


function logout() {


    /*
        Logout means the current session ends.

        Therefore everything is cleared.
    */

    answerKeyFile = null;

    textbookFile = null;


    currentStudent = {

        name: "",

        usn: "",

        answerScript: null

    };


    currentEvaluation = null;



    /*
        Reset login form.
    */

    document
        .getElementById("loginForm")
        .reset();



    /*
        Reset reference uploads.
    */

    document
        .getElementById("answerKey")
        .value = "";


    document
        .getElementById("textbook")
        .value = "";


    document
        .getElementById("answerKeyName")
        .textContent =
            "No file selected";


    document
        .getElementById("textbookName")
        .textContent =
            "No file selected";


    document
        .getElementById("referenceStatus")
        .classList
        .add("hidden");



    /*
        Return to Login.
    */

    showPage("loginPage");

}
