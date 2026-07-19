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






# ============================================================
# SHINY DASHBOARD FOR CUSTOMER CHURN ANALYTICS
# Version: 2.0 (Fixed)
# ============================================================

library(shiny)
library(DT)
library(plotly)
library(tidyverse)
library(randomForest)
library(caret)
library(janitor)

# ============================================================
# DATA PREPARATION
# ============================================================

# Load and clean data
telco_data <- read_csv("data.csv") %>%
  clean_names()

telco_clean <- telco_data %>%
  mutate(
    total_charges = as.numeric(total_charges),
    total_charges = replace_na(total_charges, 0),
    churn = factor(churn, levels = c("No", "Yes"), labels = c("No_Churn", "Churn"))
  ) %>%
  select(-customer_id)

# Prepare data for modeling
model_data <- telco_clean %>%
  mutate(across(where(is.character), as.factor))

# Train the model
set.seed(123)
train_index <- createDataPartition(model_data$churn, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

churn_model <- randomForest(
  churn ~ ., 
  data = train_data, 
  ntree = 200,
  importance = TRUE,
  na.action = na.roughfix
)

# Generate predictions and probabilities (FIXED VERSION)
churn_probs <- predict(churn_model, model_data, type = "prob")

model_data_scored <- model_data %>%
  mutate(
    churn_probability = churn_probs[, "Churn"],  # FIXED: Use matrix indexing
    predicted_churn = predict(churn_model, model_data)
  )

# ============================================================
# SHINY UI
# ============================================================

ui <- fluidPage(
  titlePanel("Customer Churn Analytics Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Filters"),
      
      selectInput("contract_filter", "Contract Type:",
                  choices = c("All", unique(telco_clean$contract)),
                  selected = "All"),
      
      sliderInput("tenure_range", "Tenure (Months):",
                  min = 0, max = 72, value = c(0, 72)),
      
      sliderInput("probability_threshold", "Churn Probability Threshold:",
                  min = 0.5, max = 0.95, value = 0.70, step = 0.05),
      
      hr(),
      p("Built with Shiny & R"),
      p("Data: Telco Customer Churn")
    ),
    
    mainPanel(
      tabsetPanel(
        
        # TAB 1: OVERVIEW
        tabPanel("Overview",
                 fluidRow(
                   valueBoxOutput("total_customers"),
                   valueBoxOutput("churn_rate"),
                   valueBoxOutput("high_risk_count")
                 ),
                 hr(),
                 fluidRow(
                   column(6, plotlyOutput("plot_contract")),
                   column(6, plotlyOutput("plot_tenure"))
                 ),
                 fluidRow(
                   column(12, plotlyOutput("plot_charges"))
                 )
        ),
        
        # TAB 2: MODEL PERFORMANCE
        tabPanel("Model Performance",
                 h3("Confusion Matrix"),
                 tableOutput("conf_matrix"),
                 hr(),
                 h3("Key Metrics"),
                 verbatimTextOutput("model_metrics"),
                 hr(),
                 h3("Variable Importance"),
                 plotlyOutput("plot_importance")
        ),
        
        # TAB 3: HIGH-RISK CUSTOMERS
        tabPanel("High-Risk Customers",
                 h3("Customers with High Churn Probability"),
                 p("Filter and export the list of customers at risk of churning."),
                 hr(),
                 DTOutput("high_risk_table"),
                 hr(),
                 downloadButton("download_high_risk", "Download CSV")
        ),
        
        # TAB 4: INTERACTIVE ANALYSIS
        tabPanel("Interactive Analysis",
                 h3("Interactive Scatter Plot"),
                 p("Hover over points to see details. Use zoom and pan."),
                 plotlyOutput("interactive_scatter", height = "600px")
        )
      )
    )
  )
)

# ============================================================
# SHINY SERVER
# ============================================================

server <- function(input, output, session) {
  
  # Reactive filtered data
  filtered_data <- reactive({
    data <- model_data_scored
    
    if (input$contract_filter != "All") {
      data <- data %>% filter(contract == input$contract_filter)
    }
    
    data <- data %>% filter(tenure >= input$tenure_range[1] & tenure <= input$tenure_range[2])
    
    return(data)
  })
  
  # High-risk customers based on threshold
  high_risk_data <- reactive({
    filtered_data() %>%
      filter(churn_probability >= input$probability_threshold) %>%
      arrange(desc(churn_probability))
  })
  
  # VALUE BOXES
  output$total_customers <- renderValueBox({
    valueBox(
      nrow(filtered_data()),
      "Total Customers",
      icon = icon("users"),
      color = "blue"
    )
  })
  
  output$churn_rate <- renderValueBox({
    rate <- mean(filtered_data()$churn == "Churn") * 100
    valueBox(
      paste0(round(rate, 1), "%"),
      "Churn Rate",
      icon = icon("chart-line"),
      color = "red"
    )
  })
  
  output$high_risk_count <- renderValueBox({
    valueBox(
      nrow(high_risk_data()),
      "High-Risk Customers",
      icon = icon("exclamation-triangle"),
      color = "orange"
    )
  })
  
  # PLOTS
  output$plot_contract <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = contract, fill = churn)) +
      geom_bar(position = "fill") +
      scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = "Churn Rate by Contract", x = "Contract", y = "Proportion") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$plot_tenure <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = tenure, fill = churn)) +
      geom_density(alpha = 0.6) +
      scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = "Tenure Distribution", x = "Tenure (Months)", y = "Density") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$plot_charges <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = monthly_charges, y = total_charges, color = churn)) +
      geom_point(alpha = 0.5) +
      scale_color_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = "Monthly vs Total Charges", x = "Monthly Charges", y = "Total Charges") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$plot_importance <- renderPlotly({
    var_imp <- importance(churn_model)
    var_imp_df <- as.data.frame(var_imp) %>%
      rownames_to_column("Variable") %>%
      arrange(desc(MeanDecreaseGini)) %>%
      head(10)
    
    p <- ggplot(var_imp_df, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_bar(stat = "identity", fill = "#3366cc") +
      coord_flip() +
      labs(title = "Top 10 Variables Driving Churn", x = "Variable", y = "Importance") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$interactive_scatter <- renderPlotly({
    plot_ly(
      data = filtered_data(),
      x = ~monthly_charges,
      y = ~total_charges,
      color = ~churn,
      colors = c("No_Churn" = "#2ca02c", "Churn" = "#d62728"),
      text = ~paste(
        "Contract:", contract,
        "<br>Tenure:", tenure, "months",
        "<br>Churn Prob:", round(churn_probability * 100, 1), "%"
      ),
      hoverinfo = "text",
      marker = list(size = 8, opacity = 0.6)
    ) %>%
      layout(
        title = "Interactive: Monthly vs Total Charges",
        xaxis = list(title = "Monthly Charges"),
        yaxis = list(title = "Total Charges")
      )
  })
  
  # MODEL PERFORMANCE
  output$conf_matrix <- renderTable({
    predictions <- predict(churn_model, test_data)
    conf_matrix <- confusionMatrix(predictions, test_data$churn, positive = "Churn")
    as.data.frame(conf_matrix$table)
  })
  
  output$model_metrics <- renderPrint({
    predictions <- predict(churn_model, test_data)
    conf_matrix <- confusionMatrix(predictions, test_data$churn, positive = "Churn")
    
    cat("=== MODEL PERFORMANCE METRICS ===\n\n")
    cat("Accuracy:", round(conf_matrix$overall['Accuracy'], 4), "\n")
    cat("Precision:", round(conf_matrix$byClass['Pos Pred Value'], 4), "\n")
    cat("Recall (Sensitivity):", round(conf_matrix$byClass['Sensitivity'], 4), "\n")
    cat("F1-Score:", round(conf_matrix$byClass['F1'], 4), "\n")
  })
  
  # HIGH-RISK TABLE
  output$high_risk_table <- renderDT({
    datatable(
      high_risk_data() %>%
        select(churn_probability, contract, tenure, monthly_charges, total_charges) %>%
        mutate(churn_probability = round(churn_probability * 100, 1)),
      options = list(pageLength = 10, scrollX = TRUE),
      caption = "High-Risk Customers List"
    ) %>%
      formatPercentage("churn_probability", 1) %>%
      formatCurrency(c("monthly_charges", "total_charges"), "$")
  })
  
  # DOWNLOAD HANDLER
  output$download_high_risk <- downloadHandler(
    filename = function() {
      paste("high_risk_customers_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write_csv(high_risk_data(), file)
    }
  )
}

# ============================================================
# RUN THE APP
# ============================================================

shinyApp(ui = ui, server = server)