import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CSV_FILE = 'expenses.csv'
HEADERS = ['Date', 'Category', 'Description', 'Amount']

def initialize_csv():
    try:
        with open(CSV_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
    except FileExistsError:
        pass 

def add_expense(category, description, amount):
    """Adds a new expense to the CSV file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, category, description, amount])
    print("Expense added successfully.")

def view_expenses():
   
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            print("\n--- All Expenses ---")
            for row in reader:
                print(f"Date: {row[0]}, Category: {row[1]}, Description: {row[2]}, Amount: ${float(row[3]):.2f}")
    except FileNotFoundError:
        print("No expenses found. The file 'expenses.csv' does not exist yet.")

def get_summary():
   
    summary = {}
    total_expenses = 0
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                category = row[1]
                amount = float(row[3])
                total_expenses += amount
                summary[category] = summary.get(category, 0) + amount
    except FileNotFoundError:
        print("No expenses found to summarize.")
        return

    print("\n--- Expense Summary ---")
    for category, amount in summary.items():
        print(f"{category}: ${amount:.2f}")
    print(f"\nTotal Expenses: ${total_expenses:.2f}")

    if summary:
        plt.figure(figsize=(6, 6))
        plt.pie(summary.values(), labels=summary.keys(), autopct='%1.1f%%', startangle=90)
        plt.title("Expense Distribution by Category")
        plt.show()

def predict_next_month_expenses():
   
    try:
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_expenses = df.groupby('Month')['Amount'].sum().reset_index()
        
        if len(monthly_expenses) < 2:
            print("Insufficient data for prediction. At least two months of data are required.")
            return

        X = np.array(range(len(monthly_expenses))).reshape(-1, 1)
        y = monthly_expenses['Amount'].values
        model = LinearRegression()
        model.fit(X, y)
        
        next_month = np.array([[len(monthly_expenses)]])
        predicted_expense = model.predict(next_month)[0]
        
        print(f"\n--- Predicted Next Month's Expenses ---")
        print(f"Based on historical data, next month's expenses are predicted to be: ${predicted_expense:.2f}")
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(monthly_expenses)), y, marker='o', label='Historical Expenses')
        plt.plot(len(monthly_expenses), predicted_expense, marker='*', color='red', label='Predicted Expense')
        plt.xlabel('Month Index')
        plt.ylabel('Total Expenses ($)')
        plt.title('Monthly Expenses Trend with Prediction')
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print("No expenses found for prediction.")

def detect_anomalies():
   
    try:
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        features = df[['Amount']].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly'] = iso_forest.fit_predict(scaled_features)
        
        anomalies = df[df['Anomaly'] == -1]
        if not anomalies.empty:
            print("\n--- Detected Anomalies in Spending ---")
            for _, row in anomalies.iterrows():
                print(f"Date: {row['Date']}, Category: {row['Category']}, Description: {row['Description']}, Amount: ${row['Amount']:.2f}")
        else:
            print("\nNo anomalies detected in spending patterns.")
        
        plt.figure(figsize=(8, 5))
        plt.scatter(df.index, df['Amount'], c=df['Anomaly'], cmap='coolwarm', label='Normal (1) / Anomaly (-1)')
        plt.xlabel('Transaction Index')
        plt.ylabel('Amount ($)')
        plt.title('Spending Patterns with Anomaly Detection')
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print("No expenses found for anomaly detection.")

def suggest_budget():
   
    try:
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        category_averages = df.groupby(['Month', 'Category'])['Amount'].sum().groupby('Category').mean().reset_index()
        
        if category_averages.empty:
            print("No data available to suggest budgets.")
            return

        print("\n--- Suggested Monthly Budgets ---")
        for _, row in category_averages.iterrows():
            category = row['Category']
            avg_amount = row['Amount']
            suggested_budget = avg_amount * 1.1  # Suggest 10% above average for flexibility
            print(f"{category}: ${suggested_budget:.2f}")
        
        plt.figure(figsize=(8, 5))
        plt.bar(category_averages['Category'], category_averages['Amount'])
        plt.xlabel('Category')
        plt.ylabel('Average Monthly Spending ($)')
        plt.title('Suggested Budgets per Category')
        plt.xticks(rotation=45)
        plt.show()

    except FileNotFoundError:
        print("No expenses found for budget suggestions.")

def generate_insights():
   
    try:
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        
       
        latest_month = df['Month'].max()
        latest_expenses = df[df['Month'] == latest_month].groupby('Category')['Amount'].sum()
        prev_month = latest_month - 1
        prev_expenses = df[df['Month'] == prev_month].groupby('Category')['Amount'].sum()
        
        print("\n--- Spending Insights ---")
        for category in latest_expenses.index:
            current = latest_expenses.get(category, 0)
            previous = prev_expenses.get(category, 0)
            if previous > 0:
                change = ((current - previous) / previous) * 100
                if change > 20:
                    print(f"You spent {change:.1f}% more on {category} this month compared to last month.")
                elif change < -20:
                    print(f"You spent {abs(change):.1f}% less on {category} this month compared to last month.")
        
        
        total_current = latest_expenses.sum()
        total_prev = prev_expenses.sum()
        if total_prev > 0 and (total_current / total_prev) > 1.2:
            print("Your overall spending increased significantly this month. Consider reviewing your expenses.")

    except FileNotFoundError:
        print("No expenses found to generate insights.")

def main():
    initialize_csv()
    while True:
        print("\n--- Expense Tracker ---")
        print("1. Add a new expense")
        print("2. View all expenses")
        print("3. Get a summary of expenses (with graph)")
        print("4. Predict next month's expenses")
        print("5. Detect anomalies in spending")
        print("6. Suggest monthly budgets")
        print("7. Generate spending insights")
        print("8. Exit")
        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            try:
                category = input("Enter expense category (e.g., Food, Transport): ")
                description = input("Enter a brief description: ")
                amount = float(input("Enter the amount: "))
                add_expense(category, description, amount)
            except ValueError:
                print("Invalid amount. Please enter a number.")
        elif choice == '2':
            view_expenses()
        elif choice == '3':
            get_summary()
        elif choice == '4':
            predict_next_month_expenses()
        elif choice == '5':
            detect_anomalies()
        elif choice == '6':
            suggest_budget()
        elif choice == '7':
            generate_insights()
        elif choice == '8':
            print("Exiting the expense tracker. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 8.")

if __name__ == "__main__":
    main()
