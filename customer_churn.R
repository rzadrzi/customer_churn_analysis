# Load required libraries
library(tidyverse)
library(janitor)
library(randomForest)
library(caret)
library(ggplot2)
library(patchwork)

# Read the dataset (make sure the CSV file is in your working directory)
telco_data <- read_csv("data.csv") %>%
  clean_names()

# Data cleaning and type conversion
telco_clean <- telco_data %>%
  mutate(
    total_charges = as.numeric(total_charges),
    total_charges = replace_na(total_charges, 0),
    churn = factor(churn, levels = c("No", "Yes"), labels = c("No_Churn", "Churn"))
  ) %>%
  select(-customer_id)

# Display a summary of the cleaned data structure
glimpse(telco_clean)

# 1. Overall Churn Rate (Pie Chart or Bar Chart)
churn_summary <- telco_clean %>%
  count(churn) %>%
  mutate(percentage = n / sum(n) * 100)

print(churn_summary)

# 2. Churn Rate by Contract Type (Stacked Bar Chart)
plot_contract <- telco_clean %>%
  ggplot(aes(x = contract, fill = churn)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
  labs(
    title = "Customer Churn Rate by Contract Type",
    x = "Contract Type",
    y = "Proportion",
    fill = "Churn Status"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# 3. Tenure Distribution by Churn Status (Density Plot)
plot_tenure <- telco_clean %>%
  ggplot(aes(x = tenure, fill = churn)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
  labs(
    title = "Tenure Distribution (Months) by Churn Status",
    x = "Tenure (Months)",
    y = "Density",
    fill = "Churn Status"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# 4. Monthly Charges vs Total Charges (Scatter Plot with Churn Highlight)
plot_charges <- telco_clean %>%
  ggplot(aes(x = monthly_charges, y = total_charges, color = churn)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
  labs(
    title = "Monthly Charges vs Total Charges by Churn Status",
    x = "Monthly Charges",
    y = "Total Charges",
    color = "Churn Status"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# Display all plots together using patchwork
plot_contract + plot_tenure + plot_charges + 
  plot_layout(ncol = 2, guides = "collect") & 
  theme(legend.position = "bottom")



# ============================================================
# SECTION 3: PREDICTIVE MODELING WITH RANDOM FOREST
# ============================================================

# 1. Prepare data for modeling (convert character to factor)
# Random Forest requires all inputs to be factor or numeric
model_data <- telco_clean %>%
  mutate(across(where(is.character), as.factor))

# 2. Split data into training (80%) and testing (20%) sets
set.seed(123) # For reproducibility
train_index <- createDataPartition(model_data$churn, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# 3. Train the Random Forest model
churn_model <- randomForest(
  churn ~ ., 
  data = train_data, 
  ntree = 200,          # Number of trees (higher = more stable)
  mtry = 4,             # Number of variables tried at each split
  importance = TRUE,    # Calculate variable importance
  na.action = na.roughfix # Handle missing values
)

# 4. Display the model summary
print(churn_model)

# 5. Evaluate the model on the test set
predictions <- predict(churn_model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$churn, positive = "Churn")

# Display confusion matrix with key metrics
print(conf_matrix)

# 6. Extract and display key performance metrics
cat("\n=== MODEL PERFORMANCE METRICS ===\n")
cat("Accuracy:", round(conf_matrix$overall['Accuracy'], 4), "\n")
cat("Precision:", round(conf_matrix$byClass['Pos Pred Value'], 4), "\n")
cat("Recall (Sensitivity):", round(conf_matrix$byClass['Sensitivity'], 4), "\n")
cat("F1-Score:", round(conf_matrix$byClass['F1'], 4), "\n")
cat("AUC (ROC):", round(conf_matrix$overall['Accuracy'], 4), "\n")

# 7. Plot Variable Importance (Top 10 predictors)
var_imp <- importance(churn_model)
var_imp_df <- as.data.frame(var_imp) %>%
  rownames_to_column("Variable") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(10)

plot_importance <- ggplot(var_imp_df, aes(x = reorder(Variable, MeanDecreaseGini), 
                                          y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "#3366cc") +
  coord_flip() +
  labs(
    title = "Top 10 Variables Driving Customer Churn",
    x = "Variable",
    y = "Importance (Mean Decrease Gini)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(plot_importance)

# 8. Generate churn probabilities for ALL customers
churn_probabilities <- predict(churn_model, model_data, type = "prob")

# 9. Add probability scores to the dataset
model_data_scored <- model_data %>%
  mutate(
    churn_probability = churn_probabilities$Churn,
    predicted_churn = predict(churn_model, model_data)
  )

# 10. Extract high-risk customers (probability > 70%)
high_risk_customers <- model_data_scored %>%
  filter(churn_probability > 0.70) %>%
  arrange(desc(churn_probability))

cat("\n=== HIGH-RISK CUSTOMER SUMMARY ===\n")
cat("Total customers analyzed:", nrow(model_data_scored), "\n")
cat("High-risk customers (>70% probability):", nrow(high_risk_customers), "\n")
cat("Percentage of high-risk:", round(nrow(high_risk_customers) / nrow(model_data_scored) * 100, 2), "%\n")

# 11. Display top 10 highest-risk customers
cat("\n=== TOP 10 HIGHEST-RISK CUSTOMERS ===\n")
print(head(high_risk_customers %>% select(churn_probability, contract, tenure, monthly_charges, total_charges), 10))

# 12. Optional: Export high-risk list to CSV for marketing team
# write_csv(high_risk_customers, "high_risk_customers_for_marketing.csv")
# cat("\n✅ High-risk customer list exported to CSV\n")

