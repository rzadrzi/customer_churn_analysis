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