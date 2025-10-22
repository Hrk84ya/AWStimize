# Comprehensive AWS Infrastructure Example
# This file demonstrates all supported AWS services with realistic configurations

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  count = 2
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "public-subnet-${count.index + 1}"
  }
}

resource "aws_subnet" "private" {
  count = 2
  vpc_id = aws_vpc.main.id
  cidr_block = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "private-subnet-${count.index + 1}"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "main-igw"
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id = aws_subnet.public[0].id
  
  tags = {
    Name = "main-nat"
  }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  
  tags = {
    Name = "nat-eip"
  }
}

# Security Groups
resource "aws_security_group" "web" {
  name = "web-sg"
  vpc_id = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name = "app-sg"
  vpc_id = aws_vpc.main.id
  
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }
}

resource "aws_security_group" "db" {
  name = "db-sg"
  vpc_id = aws_vpc.main.id
  
  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

# Load Balancer
resource "aws_lb" "main" {
  name               = "main-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.web.id]
  subnets           = aws_subnet.public[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "main-alb"
  }
}

resource "aws_lb_target_group" "web" {
  name     = "web-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
}

# EC2 Instances
resource "aws_instance" "web" {
  count = 2
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.large"  # Potentially oversized
  
  vpc_security_group_ids = [aws_security_group.web.id]
  subnet_id = aws_subnet.public[count.index].id
  
  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }
  
  tags = {
    Name = "web-server-${count.index + 1}"
  }
}

resource "aws_ebs_volume" "app_data" {
  availability_zone = aws_instance.web[0].availability_zone
  size              = 100
  type              = "gp2"  # Could be optimized to gp3
  encrypted         = true
  
  tags = {
    Name = "app-data"
  }
}

# RDS Database
resource "aws_db_subnet_group" "main" {
  name       = "main"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "main-db-subnet-group"
  }
}

resource "aws_rds_instance" "main" {
  identifier = "main-db"
  engine     = "mysql"
  engine_version = "8.0"
  instance_class = "db.t3.large"  # Potentially oversized
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp2"
  storage_encrypted    = true
  
  db_name  = "maindb"
  username = "admin"
  password = "changeme123!"
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Name = "main-database"
  }
}

# DynamoDB Table
resource "aws_dynamodb_table" "sessions" {
  name           = "user-sessions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "session_id"
  
  attribute {
    name = "session_id"
    type = "S"
  }
  
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
  
  tags = {
    Name = "user-sessions"
  }
}

resource "aws_dynamodb_table" "analytics" {
  name           = "analytics-data"
  billing_mode   = "PROVISIONED"
  read_capacity  = 10  # Potentially over-provisioned
  write_capacity = 10  # Potentially over-provisioned
  hash_key       = "event_id"
  
  attribute {
    name = "event_id"
    type = "S"
  }
  
  tags = {
    Name = "analytics-data"
  }
}

# ElastiCache
resource "aws_elasticache_subnet_group" "main" {
  name       = "main-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "main-redis"
  description                = "Redis cluster for caching"
  
  node_type                  = "cache.t3.medium"  # Potentially oversized
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.app.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "main-redis"
  }
}

# Lambda Functions
resource "aws_lambda_function" "api_handler" {
  filename         = "api_handler.zip"
  function_name    = "api-handler"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  
  memory_size = 1024  # Potentially oversized
  timeout     = 30
  
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.app.id]
  }
  
  environment {
    variables = {
      DB_HOST = aws_rds_instance.main.endpoint
      REDIS_HOST = aws_elasticache_replication_group.main.primary_endpoint_address
    }
  }
  
  tags = {
    Name = "api-handler"
  }
}

resource "aws_lambda_function" "data_processor" {
  filename         = "data_processor.zip"
  function_name    = "data-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "processor.handler"
  runtime         = "python3.9"
  
  memory_size = 512
  timeout     = 300
  
  tags = {
    Name = "data-processor"
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "main" {
  name        = "main-api"
  description = "Main API Gateway"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_method.proxy.resource_id
  http_method = aws_api_gateway_method.proxy.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.api_handler.invoke_arn
}

resource "aws_api_gateway_deployment" "main" {
  depends_on = [
    aws_api_gateway_integration.lambda,
  ]
  
  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = "prod"
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${aws_lb.main.name}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${aws_lb.main.name}"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
  
  tags = {
    Name = "main-cloudfront"
  }
}

# ECS Cluster and Service
resource "aws_ecs_cluster" "main" {
  name = "main-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "main-ecs-cluster"
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = "app-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"  # 1 vCPU
  memory                   = "2048"  # 2 GB
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "app"
      image = "nginx:latest"
      
      portMappings = [
        {
          containerPort = 80
          hostPort      = 80
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
  
  tags = {
    Name = "app-task-definition"
  }
}

resource "aws_ecs_service" "app" {
  name            = "app-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.web.arn
    container_name   = "app"
    container_port   = 80
  }
  
  depends_on = [aws_lb_target_group.web]
  
  tags = {
    Name = "app-ecs-service"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/app"
  retention_in_days = 7
  
  tags = {
    Name = "ecs-logs"
  }
}

# IAM Roles
resource "aws_iam_role" "lambda_role" {
  name = "lambda-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role" "ecs_execution_role" {
  name = "ecs-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}