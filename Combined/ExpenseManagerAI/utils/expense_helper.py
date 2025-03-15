import pandas as pd

# Load purchase data
data = pd.read_csv("Combined/ExpenseManagerAI/datasets/augmented_ecommerce_data.csv")
data["Purchase_Date"] = pd.to_datetime(data["Purchase_Date"])

# Helper function to calculate category spending patterns
def get_category_spending_pattern(user_id):
    user_purchases = data[data["User_ID"] == user_id]
    if user_purchases.empty:
        return {}

    # Calculate total spending per category
    spending_pattern = (
        user_purchases.groupby("Category")["Final_Price(Rs.)"].sum()
        / user_purchases["Final_Price(Rs.)"].sum()
    ).to_dict()
    return spending_pattern

# Enhanced Recommendation Logic
def recommend_products_with_pattern(user_id, budget):
    # Step 1: Filter user purchases and learn patterns
    user_purchases = data[data["User_ID"] == user_id]
    spending_pattern = get_category_spending_pattern(user_id)

    total_spent = 0
    recommended_products = []

    # Step 2: For new users, provide default products
    if not spending_pattern:
        default_products = [
            {"category": "Groceries", "price": 1200},
            {"category": "Household Essentials", "price": 1000},
            {"category": "Personal Care", "price": 800},
            {"category": "Clothing", "price": 700}
        ]
        for product in default_products:
            if total_spent + product["price"] <= budget:
                recommended_products.append(product)
                total_spent += product["price"]
        return {
            "status": "new_user",
            "message": "No purchase history found. Providing default products.",
            "recommended_products": recommended_products,
            "total_spent": total_spent,
            "remaining_budget": budget - total_spent
        }

    # Step 3: Allocate budget dynamically based on spending patterns
    for category, proportion in spending_pattern.items():
        category_budget = budget * proportion
        category_products = user_purchases[user_purchases["Category"] == category].sort_values(by="Final_Price(Rs.)")

        for _, row in category_products.iterrows():
            if total_spent + row["Final_Price(Rs.)"] <= category_budget:
                recommended_products.append({
                    "category": row["Category"],
                    "price": row["Final_Price(Rs.)"],
                    "purchase_date": row["Purchase_Date"].strftime("%Y-%m-%d")
                })
                total_spent += row["Final_Price(Rs.)"]

    # Step 4: Add products from underrepresented categories
    remaining_budget = budget - total_spent
    if remaining_budget > 0:
        unused_categories = data[~data["Category"].isin(spending_pattern.keys())]["Category"].unique()
        default_products = [
            {"category": category, "price": round(remaining_budget / 3)}
            for category in unused_categories
        ]
        for product in default_products:
            if total_spent + product["price"] <= budget:
                recommended_products.append(product)
                total_spent += product["price"]

    # Step 5: Fill gaps with dynamically chosen products
    if total_spent < 0.95 * budget:
        filler_products = [
            {"category": "Groceries", "price": 1000},
            {"category": "Personal Care", "price": 800},
            {"category": "Stationery", "price": 600}
        ]
        for product in filler_products:
            if total_spent + product["price"] <= budget:
                recommended_products.append(product)
                total_spent += product["price"]

    # Step 6: Final response
    return {
        "status": "success",
        "recommended_products": recommended_products,
        "total_spent": total_spent,
        "remaining_budget": budget - total_spent
    }
