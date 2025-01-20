### Advertising.csv를 불러와 데이터 로드하기!

advertising_data <- read.csv("Advertising.csv")
head(advertising_data)


### Multiple Linear Regression을 수행해봅시다!

model <- lm(sales ~ TV + radio + newspaper, data = advertising_data)

summary(model)



### Correlation Matrix를 만들어 출력해주세요!

# 상관행렬 계산
correlation_matrix <- cor(advertising_data[, c("TV", "radio", "newspaper", "sales")])

# 상관행렬 출력
print(correlation_matrix)



