# Summaries summary command
  # Identify missing values, will help decide whether to drop the missing rows if negligible or we need to drop columns or impute values if large number of values are missing.
  # Encoding of missing values as a unique variable

  # INVALID VALUES AND OUTLIERS
  # Negative values can be invalid, for example for income and extreme values like zero and 111 outliers for columns like age

  # DATA RANGE

usage_data = read.csv("data/UsageData.csv")

check_col_sparsity <- function(input_data_col, sparsity_threshold, col_name){
  count_of_rows <- nrow(input_data_col)
  input_data_col[is.na(input_data_col)] <- 0
  count_of_zeros <- length(input_data_col[input_data_col==0])
  per_zero <- count_of_zeros/as.double(count_of_rows)
  if(per_zero < sparsity_threshold)
  {
    return(c(FALSE, round(per_zero*100)))
  }
  else {
    return(c(TRUE, round(per_zero*100)))
  }
  
}

print(check_col_sparsity(usage_data['forms_filled'],0.5,'forms_filled'))