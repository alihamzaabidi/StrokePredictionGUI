# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Define a class to handle the Data from MongoDB to Python
class DatabaseHandler:
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client['Assessment_Data']
        self.collection = self.db['Stroke_Prediction']
# Define a funtion to retrive the Data from MongoDB collection to Python dataframe
    def retrieve_data(self):
        data = list(self.collection.find())
        return pd.DataFrame(data)

# Define a class to handle the Decision Tree model
class DecisionTreeHandler:
    def __init__(self, data):
        self.dtree = None
        self.stroke_data = data

    def train_decision_tree(self):
        stroke_data_x = self.stroke_data.drop("stroke", axis=1)
        stroke_data_y = self.stroke_data['stroke']
        stroke_data_x_encoded = pd.get_dummies(stroke_data_x, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(stroke_data_x_encoded, stroke_data_y, test_size=0.1)
        self.dtree = DecisionTreeClassifier()
        self.dtree.fit(X_train, y_train)
# Define a class to handle the GUI prediction page
class GUIHandler:
    def __init__(self, root, dtree):
        self.root = root
        self.dtree = dtree

        root.title("Stroke Prediction Page")
        root.minsize(600, 500)
        root.geometry("900x600")

        self.title_label = Label(
            text='''WELCOME TO THE STROKE PREDICTION PAGE
            A stroke is caused when blood flow to a part of the brain is stopped abruptly. 
            Without the blood supply, the brain cells gradually die, 
            and disability occurs depending on the area of the brain affected.''',
            fg="black",
            font="Georgia"
        )
        self.title_label.pack(side=BOTTOM, fill=X, padx=5, pady=10)

        self.create_login_frame()

        self.entry_sex = None
        self.entry_age = None
        self.entry_hypertension = None
        self.entry_heart_disease = None
        self.entry_ever_married = None
        self.entry_work_type = None
        self.entry_residence_type = None
        self.entry_avg_glucose_level = None
        self.entry_bmi = None
        self.entry_smoking_status = None
# Define a function to handle the login frame
    def create_login_frame(self):
        self.login_frame = Frame(self.root)
        self.login_frame.pack()
        self.username_label = Label(self.login_frame, text="Username:", fg="black")
        self.username_label.pack(anchor="w")
        self.username_entry = Entry(self.login_frame)
        self.username_entry.pack(anchor="w")

        self.password_label = Label(self.login_frame, text="Password:", fg="black")
        self.password_label.pack(anchor="w")
        self.password_entry = Entry(self.login_frame, show="*")
        self.password_entry.pack(anchor="w")

        self.login_button = Button(self.login_frame, text="Login", command=self.login)
        self.login_button.pack(anchor="w")
# Define an image to show the stroke pic
        photo = Image.open("Stroke pic.png")
        photo = ImageTk.PhotoImage(photo)
        self.picture_label = Label(self.login_frame, image=photo)
        self.picture_label.image = photo
        self.picture_label.pack(anchor=CENTER)
# Define a function to handle the login data with username and password
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username == "ali" and password == "h":
            print("Login successful!")
            self.create_main_window()
        else:
            self.show_invalid_login_message()
# Define a function to handle the invalid login message
    def show_invalid_login_message(self):
        invalid_window = Toplevel(self.root)
        invalid_window.title("Invalid Login")

        message_label = Label(invalid_window, text="Invalid username and password \n Please Check Again", font=("Helvetica", 10))
        message_label.pack(padx=20, pady=20)

        close_button = Button(invalid_window, text="Close", command=invalid_window.destroy)
        close_button.pack()
# Define a function to create a main window with decision tree and prediction button
    def create_main_window(self):
        self.login_frame.destroy()

        main_window = Toplevel(self.root)
        main_window.title("Stroke Prediction Window")
        main_window.geometry("300x600")

        decision_tree_button = Button(main_window, text="Decision Tree", command=self.open_decision_tree_window)
        decision_tree_button.pack()

        prediction_button = Button(main_window, text="Prediction", command=self.perform_classification)
        prediction_button.pack()

        Stroke_Info = Label(main_window,
            text='''IMPORTANT INFORMATION TO USE \n 1 = Yes and Male. \n 0 = No and Female''',
            fg="black",
            font="Georgia"
        )
        Stroke_Info.pack(side=BOTTOM, fill=X, padx=5, pady=10)

        label_sex = Label(main_window, text="Sex")
        label_sex.pack(anchor="w")
        self.entry_sex = Entry(main_window)
        self.entry_sex.pack(anchor="w")

        label_age = Label(main_window, text="Age")
        label_age.pack(anchor="w")
        self.entry_age = Entry(main_window)
        self.entry_age.pack(anchor="w")

        label_hypertension = Label(main_window, text="Hypertension")
        label_hypertension.pack(anchor="w")
        self.entry_hypertension = Entry(main_window)
        self.entry_hypertension.pack(anchor="w")

        label_heart_disease = Label(main_window, text="Heart Disease")
        label_heart_disease.pack(anchor="w")
        self.entry_heart_disease = Entry(main_window)
        self.entry_heart_disease.pack(anchor="w")

        label_ever_married = Label(main_window, text="Ever Married")
        label_ever_married.pack(anchor="w")
        self.entry_ever_married = Entry(main_window)
        self.entry_ever_married.pack(anchor="w")

        label_work_type = Label(main_window, text="Work Type")
        label_work_type.pack(anchor="w")
        self.entry_work_type = Entry(main_window)
        self.entry_work_type.pack(anchor="w")

        label_residence_type = Label(main_window, text="Residence Type")
        label_residence_type.pack(anchor="w")
        self.entry_residence_type = Entry(main_window)
        self.entry_residence_type.pack(anchor="w")

        label_avg_glucose_level = Label(main_window, text="Avg Glucose Level")
        label_avg_glucose_level.pack(anchor="w")
        self.entry_avg_glucose_level = Entry(main_window)
        self.entry_avg_glucose_level.pack(anchor="w")

        label_bmi = Label(main_window, text="BMI")
        label_bmi.pack(anchor="w")
        self.entry_bmi = Entry(main_window)
        self.entry_bmi.pack(anchor="w")

        label_smoking_status = Label(main_window, text="Smoking Status")
        label_smoking_status.pack(anchor="w")
        self.entry_smoking_status = Entry(main_window)
        self.entry_smoking_status.pack(anchor="w")
# Define a function to open a decision tree in seprate window
    def open_decision_tree_window(self):
        decision_tree_window = Toplevel(self.root)
        decision_tree_window.title("Decision Tree Window")

        image = Image.open('decision_tree_image.png')
        photo = ImageTk.PhotoImage(image)

        label = Label(decision_tree_window, image=photo)
        label.image = photo
        label.pack()
# Define a function to open a prediction classification through MongoDB
    def perform_classification(self):
        # Connect to the MongoDB server
        client = MongoClient('mongodb://localhost:27017/')

        # Access the database and collection
        db = client['Assessment_Data']
        collection = db['Stroke_Prediction']

        # Retrieve data from MongoDB and convert it to a DataFrame, excluding the '_id' column
        data = list(collection.find({}, {"_id": 0}))

        stroke_data = pd.DataFrame(data)

        # Check for missing values
        stroke_data.isnull().sum()

        # Now the process of Train and Test, then Split
        x = stroke_data.drop(['stroke'], axis=1)
        y = stroke_data['stroke']

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train the K-Nearest Neighbors (KNN) model
        kn = KNeighborsClassifier(n_neighbors=10)
        kn.fit(xtrain, ytrain)
        score_kn = kn.score(xtest, ytest)

        # As an example, use 'a' list to test the KNN model
        #a = np.array([self.entry_sex.get(), self.entry_age.get(), self.entry_hypertension.get(), self.entry_heart_disease.get(), self.entry_ever_married.get(), self.entry_work_type.get(), self.entry_residence_type.get(), self.entry_avg_glucose_level.get(), self.entry_bmi.get(), self.entry_smoking_status.get()])
        
        # For Example this data can be use
        a = np.array([[1, 63, 0, 1, 1, 4, 1, 228.69, 36.6, 1]])
        
        prediction_kn = kn.predict(a)


        # Define a function to show the prediction result
        def prediction():
            result_window = Toplevel(self.root)
            result_window.title("Prediction Result")

            if prediction_kn[0] == 1:
                result_label = Label(result_window, text='Chances of stroke are high', font=("Helvetica", 10))
                result_label.pack(padx=20, pady=20)
            else:
                result_label = Label(result_window, text='Chances of stroke are low', font=("Helvetica", 10))
                result_label.pack(padx=20, pady=20)

        # Call the prediction function
        prediction()

# Define a class to open a prediction handeling window
class PredictionHandler:
    def __init__(self, data):
        self.data = data
# Train the K-Nearest Neighbors (KNN) model
    def train_knn_model(self):
        x = self.data.drop(['stroke'], axis=1)
        y = self.data['stroke']
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
        self.knn_model = KNeighborsClassifier(n_neighbors=10)
        self.knn_model.fit(xtrain, ytrain)

# Define a funtion to run the prdiction model
    def predict_stroke(self, attributes):
        prediction_kn = self.knn_model.predict(attributes)
        return prediction_kn
    
# Define a funtion to result the prdiction model
    def show_prediction_result(self, result):
        result_window = Toplevel(self.root)
        result_window.title("Prediction Result")

        if result == 1:
            result_label = Label(result_window, text='Chances of stroke are high', font=("Helvetica", 10))
            result_label.pack(padx=20, pady=20)
        else:
            result_label = Label(result_window, text='Chances of stroke are low', font=("Helvetica", 10))
            result_label.pack(padx=20, pady=20)


# Entry point of the program
if __name__ == "__main__":
    root = Tk()
    db_handler = DatabaseHandler('mongodb://localhost:27017/')
    data = db_handler.retrieve_data()
    dtree_handler = DecisionTreeHandler(data)
    dtree_handler.train_decision_tree()
    app = GUIHandler(root, dtree_handler)
    root.mainloop()
