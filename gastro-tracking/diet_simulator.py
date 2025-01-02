from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DietSimulator:
    def __init__(self, base_foods, special_foods, start_date="12-31-2025"):
        """
        Initialize the diet simulator with base foods and special foods.

        Args:
        - base_foods (dict): A dictionary where keys are food names and values are probabilities of inclusion in meals.
        - special_foods (dict): A dictionary where keys are food names and values are dictionaries of effects
                                (e.g., {"Fatigue": effect_strength, "Throat Itching": effect_strength}).
        - start_date (str): The starting date for the simulation (format: "MM-DD-YYYY").
        """
        self.base_foods = base_foods
        self.special_foods = special_foods
        self.start_date = datetime.strptime(start_date, "%m-%d-%Y")

    def generate_meals(self, days):
        """
        Generate meals for the given number of days.

        Args:
        - days (int): Number of days to simulate.

        Returns:
        - pd.DataFrame: A DataFrame containing meal data.
        """
        all_foods = {**self.base_foods, **{k: v["probability"] for k, v in self.special_foods.items()}}
        meal_types = ["B", "L", "D", "S"]
        data = []
        
        for day_offset in range(days):
            current_date = self.start_date + timedelta(days=day_offset)
            for meal_type in meal_types[:3]:  # Breakfast, Lunch, Dinner
                meal_foods = [food for food, prob in all_foods.items() if np.random.rand() < prob]
                data.append({
                    "Date": current_date,
                    "Meal": meal_type,
                    "Foods": ", ".join(meal_foods)
                })
            if np.random.rand() < 0.5:  # Randomly include a snack
                meal_foods = [food for food, prob in all_foods.items() if np.random.rand() < prob]
                data.append({
                    "Date": current_date,
                    "Meal": "S",
                    "Foods": ", ".join(meal_foods)
                })
        return pd.DataFrame(data)

    def generate_outcomes(self, meal_data):
        """
        Generate outcomes based on the meals and food effects.

        Args:
        - meal_data (pd.DataFrame): A DataFrame containing meal data.

        Returns:
        - pd.DataFrame: A DataFrame containing meal data with outcomes.
        """
        outcome_effects = {effect: [] for effect in {key for v in self.special_foods.values() for key in v["effects"]}}
        
        for _, row in meal_data.iterrows():
            foods = row["Foods"].split(", ") if row["Foods"] else []
            for outcome in outcome_effects:
                # Check if any food in the meal independently triggers the outcome
                outcome_effects[outcome].append(
                    any(np.random.rand() < self.special_foods[food]["effects"][outcome] 
                        for food in foods if food in self.special_foods and outcome in self.special_foods[food]["effects"])
                )
        
        # Add outcomes to the meal data
        for outcome, outcome_values in outcome_effects.items():
            meal_data[outcome] = outcome_values
        return meal_data

    def simulate(self, days=30):
        """
        Simulate dietary intake and symptom data.

        Args:
        - days (int): Number of days to simulate.

        Returns:
        - pd.DataFrame: A DataFrame containing simulated data with meals and outcomes.
        """
        meal_data = self.generate_meals(days)
        meal_data_with_outcomes = self.generate_outcomes(meal_data)
        return meal_data_with_outcomes
