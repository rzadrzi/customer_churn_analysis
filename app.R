# ============================================================
# SHINY DASHBOARD FOR CUSTOMER CHURN ANALYTICS
# ============================================================

# Load all required libraries
library(shiny)
library(shinydashboard)  # This is critical for valueBox
library(DT)
library(plotly)
library(tidyverse)
library(randomForest)
library(caret)
library(janitor)

# ============================================================
# DATA PREPARATION
# ============================================================

telco_data <- read_csv("data.csv") %>%
  clean_names()

telco_clean <- telco_data %>%
  mutate(
    total_charges = as.numeric(total_charges),
    total_charges = replace_na(total_charges, 0),
    churn = factor(churn, levels = c("No", "Yes"), labels = c("No_Churn", "Churn"))
  ) %>%
  select(-customer_id)

model_data <- telco_clean %>%
  mutate(across(where(is.character), as.factor))

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

churn_probs <- predict(churn_model, model_data, type = "prob")

model_data_scored <- model_data %>%
  mutate(
    churn_probability = churn_probs[, "Churn"],
    predicted_churn = predict(churn_model, model_data)
  )

# ============================================================
# UI
# ============================================================

ui <- dashboardPage(
  dashboardHeader(title = "Churn Analytics"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("Model Performance", tabName = "model", icon = icon("cogs")),
      menuItem("High-Risk Customers", tabName = "highrisk", icon = icon("exclamation-triangle")),
      menuItem("Interactive Analysis", tabName = "interactive", icon = icon("chart-line")),
      hr(),
      h4("Filters", style = "color: white; padding-left: 15px;"),
      
      selectInput("contract_filter", "Contract Type:",
                  choices = c("All", unique(telco_clean$contract)),
                  selected = "All"),
      
      sliderInput("tenure_range", "Tenure (Months):",
                  min = 0, max = 72, value = c(0, 72)),
      
      sliderInput("probability_threshold", "Churn Probability:",
                  min = 0.5, max = 0.95, value = 0.70, step = 0.05)
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(tabName = "overview",
              fluidRow(
                valueBoxOutput("total_customers", width = 4),
                valueBoxOutput("churn_rate", width = 4),
                valueBoxOutput("high_risk_count", width = 4)
              ),
              fluidRow(
                box(title = "Churn Rate by Contract", status = "primary", solidHeader = TRUE,
                    width = 6, plotlyOutput("plot_contract")),
                box(title = "Tenure Distribution", status = "primary", solidHeader = TRUE,
                    width = 6, plotlyOutput("plot_tenure"))
              ),
              fluidRow(
                box(title = "Monthly vs Total Charges", status = "info", solidHeader = TRUE,
                    width = 12, plotlyOutput("plot_charges"), height = "500px")
              )
      ),
      
      tabItem(tabName = "model",
              fluidRow(
                box(title = "Confusion Matrix", status = "success", solidHeader = TRUE,
                    width = 6, tableOutput("conf_matrix")),
                box(title = "Key Metrics", status = "success", solidHeader = TRUE,
                    width = 6, verbatimTextOutput("model_metrics"))
              ),
              fluidRow(
                box(title = "Top 10 Variables Driving Churn", status = "warning", solidHeader = TRUE,
                    width = 12, plotlyOutput("plot_importance"), height = "500px")
              )
      ),
      
      tabItem(tabName = "highrisk",
              fluidRow(
                box(title = "High-Risk Customers List", status = "danger", solidHeader = TRUE,
                    width = 12,
                    p("Filter and export customers at risk of churning."),
                    DTOutput("high_risk_table"),
                    hr(),
                    downloadButton("download_high_risk", "Download CSV", class = "btn-primary")
                )
              )
      ),
      
      tabItem(tabName = "interactive",
              fluidRow(
                box(title = "Interactive Scatter Plot", status = "info", solidHeader = TRUE,
                    width = 12,
                    p("Hover over points to see details. Use zoom and pan."),
                    plotlyOutput("interactive_scatter", height = "600px")
                )
              )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {
  
  filtered_data <- reactive({
    data <- model_data_scored
    
    if (input$contract_filter != "All") {
      data <- data %>% filter(contract == input$contract_filter)
    }
    
    data <- data %>% filter(tenure >= input$tenure_range[1] & tenure <= input$tenure_range[2])
    
    return(data)
  })
  
  high_risk_data <- reactive({
    filtered_data() %>%
      filter(churn_probability >= input$probability_threshold) %>%
      arrange(desc(churn_probability))
  })
  
  output$total_customers <- renderValueBox({
    valueBox(
      value = nrow(filtered_data()),
      subtitle = "Total Customers",
      icon = icon("users"),
      color = "aqua"
    )
  })
  
  output$churn_rate <- renderValueBox({
    rate <- mean(filtered_data()$churn == "Churn") * 100
    valueBox(
      value = paste0(round(rate, 1), "%"),
      subtitle = "Churn Rate",
      icon = icon("chart-line"),
      color = "red"
    )
  })
  
  output$high_risk_count <- renderValueBox({
    valueBox(
      value = nrow(high_risk_data()),
      subtitle = "High-Risk Customers",
      icon = icon("exclamation-triangle"),
      color = "yellow"
    )
  })
  
  output$plot_contract <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = contract, fill = churn)) +
      geom_bar(position = "fill") +
      scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = NULL, x = "Contract", y = "Proportion") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$plot_tenure <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = tenure, fill = churn)) +
      geom_density(alpha = 0.6) +
      scale_fill_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = NULL, x = "Tenure (Months)", y = "Density") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$plot_charges <- renderPlotly({
    p <- filtered_data() %>%
      ggplot(aes(x = monthly_charges, y = total_charges, color = churn)) +
      geom_point(alpha = 0.5) +
      scale_color_manual(values = c("No_Churn" = "#2ca02c", "Churn" = "#d62728")) +
      labs(title = NULL, x = "Monthly Charges", y = "Total Charges") +
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
      labs(title = NULL, x = "Variable", y = "Importance") +
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
        title = NULL,
        xaxis = list(title = "Monthly Charges"),
        yaxis = list(title = "Total Charges")
      )
  })
  
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
# RUN APP
# ============================================================

shinyApp(ui = ui, server = server)