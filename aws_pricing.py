from typing import Dict, Optional, Union

class AWSPricing:
    def __init__(self):
        # EC2 Instance Pricing (monthly USD, us-east-1)
        self.ec2_pricing = {
            't2.nano': 4.18,
            't2.micro': 8.35,
            't2.small': 16.70,
            't2.medium': 33.41,
            't2.large': 66.82,
            't2.xlarge': 133.65,
            't2.2xlarge': 267.30,
            't3.nano': 3.80,
            't3.micro': 7.59,
            't3.small': 15.18,
            't3.medium': 30.37,
            't3.large': 60.74,
            't3.xlarge': 121.47,
            't3.2xlarge': 242.95,
            't3a.nano': 3.42,
            't3a.micro': 6.84,
            't3a.small': 13.68,
            't3a.medium': 27.36,
            't3a.large': 54.72,
            't3a.xlarge': 109.44,
            't3a.2xlarge': 218.88,
            'm5.large': 69.35,
            'm5.xlarge': 138.70,
            'm5.2xlarge': 277.40,
            'm5.4xlarge': 554.80,
            'c5.large': 61.32,
            'c5.xlarge': 122.64,
            'c5.2xlarge': 245.28,
            'c5.4xlarge': 490.56,
            'r5.large': 90.25,
            'r5.xlarge': 180.50,
            'r5.2xlarge': 361.00,
            'r5.4xlarge': 722.00,
        }
        
        # RDS Instance Pricing (monthly USD)
        self.rds_pricing = {
            'db.t3.micro': 14.60,
            'db.t3.small': 29.20,
            'db.t3.medium': 58.40,
            'db.t3.large': 116.80,
            'db.t3.xlarge': 233.60,
            'db.t3.2xlarge': 467.20,
            'db.t4g.micro': 12.41,
            'db.t4g.small': 24.82,
            'db.t4g.medium': 49.64,
            'db.t4g.large': 99.28,
            'db.m5.large': 131.40,
            'db.m5.xlarge': 262.80,
            'db.m5.2xlarge': 525.60,
            'db.r5.large': 180.50,
            'db.r5.xlarge': 361.00,
            'db.r5.2xlarge': 722.00,
        }
        
        # EBS Storage Pricing (per GB per month)
        self.storage_pricing = {
            'gp2': 0.10,
            'gp3': 0.08,
            'io1': 0.125,
            'io2': 0.125,
            'st1': 0.045,
            'sc1': 0.025,
            'standard': 0.05,
        }
        
        # Lambda Pricing (per million requests and GB-seconds)
        self.lambda_pricing = {
            'requests_per_million': 0.20,
            'gb_seconds': 0.0000166667,  # per GB-second
            'free_tier_requests': 1000000,  # 1M requests per month
            'free_tier_gb_seconds': 400000,  # 400K GB-seconds per month
        }
        
        # API Gateway Pricing (per million requests)
        self.api_gateway_pricing = {
            'rest_api_requests': 3.50,  # per million requests
            'http_api_requests': 1.00,  # per million requests
            'websocket_messages': 1.00,  # per million messages
            'websocket_connections': 0.25,  # per million connection minutes
        }
        
        # CloudFront Pricing (per GB and per 10K requests)
        self.cloudfront_pricing = {
            'data_transfer_out_gb': 0.085,  # first 10TB per month
            'requests_per_10k': 0.0075,  # HTTP requests
            'https_requests_per_10k': 0.0100,  # HTTPS requests
            'origin_requests_per_10k': 0.0050,  # Origin requests
        }
        
        # ELB/ALB Pricing (per hour and per LCU)
        self.elb_pricing = {
            'classic_lb_hour': 18.25,  # per month (24/7)
            'application_lb_hour': 16.43,  # per month (24/7)
            'network_lb_hour': 16.43,  # per month (24/7)
            'gateway_lb_hour': 16.43,  # per month (24/7)
            'alb_lcu_hour': 5.84,  # per LCU per month
            'nlb_lcu_hour': 4.38,  # per LCU per month
            'gwlb_lcu_hour': 2.92,  # per LCU per month
        }
        
        # ECS/EKS/Fargate Pricing
        self.container_pricing = {
            'eks_cluster_hour': 73.00,  # per cluster per month
            'fargate_vcpu_hour': 29.565,  # per vCPU per month
            'fargate_memory_gb_hour': 3.24,  # per GB per month
            'ecs_ec2_free': 0.0,  # ECS on EC2 is free, pay for EC2 instances
        }
        
        # DynamoDB Pricing
        self.dynamodb_pricing = {
            'on_demand_read_units': 0.25,  # per million read request units
            'on_demand_write_units': 1.25,  # per million write request units
            'provisioned_read_units': 0.09,  # per RCU per month
            'provisioned_write_units': 0.47,  # per WCU per month
            'storage_gb': 0.25,  # per GB per month
            'backup_gb': 0.20,  # per GB per month
            'global_tables_replicated_write': 1.875,  # per million replicated write units
        }
        
        # ElastiCache Pricing
        self.elasticache_pricing = {
            'cache.t3.micro': 11.68,
            'cache.t3.small': 23.36,
            'cache.t3.medium': 46.72,
            'cache.t4g.nano': 5.11,
            'cache.t4g.micro': 10.22,
            'cache.t4g.small': 20.44,
            'cache.t4g.medium': 40.88,
            'cache.m5.large': 109.50,
            'cache.m5.xlarge': 219.00,
            'cache.m5.2xlarge': 438.00,
            'cache.r5.large': 153.30,
            'cache.r5.xlarge': 306.60,
            'cache.r5.2xlarge': 613.20,
        }
        
        # NAT Gateway Pricing
        self.nat_gateway_pricing = {
            'hourly': 32.85,  # per month (24/7)
            'data_processing_gb': 0.045,  # per GB processed
        }
        
        # Elastic IP Pricing
        self.elastic_ip_pricing = {
            'unused_ip': 3.65,  # per unused IP per month
            'additional_ip': 3.65,  # per additional IP per instance per month
        }
    
    def get_ec2_cost(self, instance_type: str) -> float:
        return self.ec2_pricing.get(instance_type, 50.0)
    
    def get_rds_cost(self, instance_class: str) -> float:
        return self.rds_pricing.get(instance_class, 100.0)
    
    def calculate_resource_cost(self, resource_type: str, config: Dict) -> float:
        if resource_type == 'aws_instance':
            instance_type = config.get('instance_type', 't3.micro')
            return self.get_ec2_cost(instance_type)
        
        elif resource_type == 'aws_rds_instance':
            instance_class = config.get('instance_class', 'db.t3.micro')
            base_cost = self.get_rds_cost(instance_class)
            
            storage_size = config.get('allocated_storage', 20)
            storage_type = config.get('storage_type', 'gp2')
            storage_cost = storage_size * self.storage_pricing.get(storage_type, 0.10)
            
            return base_cost + storage_cost
        
        elif resource_type == 'aws_s3_bucket':
            return 5.0  # Basic storage estimate
        
        elif resource_type == 'aws_lambda_function':
            return self._calculate_lambda_cost(config)
        
        elif resource_type == 'aws_api_gateway_rest_api':
            return self._calculate_api_gateway_cost(config, 'rest')
        
        elif resource_type == 'aws_apigatewayv2_api':
            return self._calculate_api_gateway_cost(config, 'http')
        
        elif resource_type == 'aws_cloudfront_distribution':
            return self._calculate_cloudfront_cost(config)
        
        elif resource_type in ['aws_lb', 'aws_alb', 'aws_elb']:
            return self._calculate_load_balancer_cost(config)
        
        elif resource_type == 'aws_ebs_volume':
            return self._calculate_ebs_cost(config)
        
        elif resource_type == 'aws_eip':
            return self._calculate_elastic_ip_cost(config)
        
        elif resource_type == 'aws_nat_gateway':
            return self._calculate_nat_gateway_cost(config)
        
        elif resource_type == 'aws_ecs_cluster':
            return 0.0  # ECS cluster itself is free
        
        elif resource_type == 'aws_ecs_service':
            return self._calculate_ecs_service_cost(config)
        
        elif resource_type == 'aws_eks_cluster':
            return self.container_pricing['eks_cluster_hour']
        
        elif resource_type == 'aws_ecs_task_definition':
            return self._calculate_fargate_cost(config)
        
        elif resource_type == 'aws_dynamodb_table':
            return self._calculate_dynamodb_cost(config)
        
        elif resource_type == 'aws_elasticache_cluster':
            return self._calculate_elasticache_cost(config)
        
        elif resource_type == 'aws_elasticache_replication_group':
            return self._calculate_elasticache_replication_cost(config)
        
        else:
            return 0.0
    
    def get_rightsizing_recommendation(self, resource_type: str, current_config: Dict) -> Optional[Dict]:
        if resource_type == 'aws_instance':
            return self._get_ec2_rightsizing(current_config)
        
        elif resource_type == 'aws_rds_instance':
            return self._get_rds_rightsizing(current_config)
        
        elif resource_type == 'aws_lambda_function':
            return self._get_lambda_rightsizing(current_config)
        
        elif resource_type == 'aws_elasticache_cluster':
            return self._get_elasticache_rightsizing(current_config)
        
        elif resource_type == 'aws_ebs_volume':
            return self._get_ebs_rightsizing(current_config)
        
        elif resource_type == 'aws_dynamodb_table':
            return self._get_dynamodb_rightsizing(current_config)
        
        return None
    
    def _get_ec2_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        current_type = current_config.get('instance_type', '')
        current_cost = self.get_ec2_cost(current_type)
        
        alternatives = {
            't3.2xlarge': 't3.xlarge',
            't3.xlarge': 't3.large',
            't3.large': 't3.medium', 
            't3.medium': 't3.small',
            't2.2xlarge': 't2.xlarge',
            't2.xlarge': 't2.large',
            't2.large': 't2.medium',
            't2.medium': 't2.small',
            'm5.4xlarge': 'm5.2xlarge',
            'm5.2xlarge': 'm5.xlarge',
            'm5.xlarge': 'm5.large',
            'c5.4xlarge': 'c5.2xlarge',
            'c5.2xlarge': 'c5.xlarge',
            'c5.xlarge': 'c5.large',
            'r5.4xlarge': 'r5.2xlarge',
            'r5.2xlarge': 'r5.xlarge',
            'r5.xlarge': 'r5.large',
        }
        
        if current_type in alternatives:
            suggested_type = alternatives[current_type]
            suggested_cost = self.get_ec2_cost(suggested_type)
            savings = ((current_cost - suggested_cost) / current_cost) * 100
            
            return {
                'current_type': current_type,
                'suggested_type': suggested_type,
                'current_cost': current_cost,
                'suggested_cost': suggested_cost,
                'monthly_savings': round(current_cost - suggested_cost, 2),
                'savings_percentage': round(savings, 1)
            }
        return None
    
    def _get_rds_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        current_class = current_config.get('instance_class', '')
        current_cost = self.get_rds_cost(current_class)
        
        alternatives = {
            'db.t3.2xlarge': 'db.t3.xlarge',
            'db.t3.xlarge': 'db.t3.large',
            'db.t3.large': 'db.t3.medium',
            'db.t3.medium': 'db.t3.small',
            'db.m5.2xlarge': 'db.m5.xlarge',
            'db.m5.xlarge': 'db.m5.large',
            'db.r5.2xlarge': 'db.r5.xlarge',
            'db.r5.xlarge': 'db.r5.large',
        }
        
        if current_class in alternatives:
            suggested_class = alternatives[current_class]
            suggested_cost = self.get_rds_cost(suggested_class)
            savings = ((current_cost - suggested_cost) / current_cost) * 100
            
            return {
                'current_type': current_class,
                'suggested_type': suggested_class,
                'current_cost': current_cost,
                'suggested_cost': suggested_cost,
                'monthly_savings': round(current_cost - suggested_cost, 2),
                'savings_percentage': round(savings, 1)
            }
        return None
    
    def _get_lambda_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        current_memory = current_config.get('memory_size', 128)
        
        # Suggest smaller memory if using default high values
        if current_memory >= 1024:
            suggested_memory = max(512, current_memory // 2)
            current_cost = self._calculate_lambda_cost(current_config)
            
            suggested_config = current_config.copy()
            suggested_config['memory_size'] = suggested_memory
            suggested_cost = self._calculate_lambda_cost(suggested_config)
            
            if suggested_cost < current_cost:
                savings = current_cost - suggested_cost
                savings_pct = (savings / current_cost) * 100
                
                return {
                    'current_type': f'{current_memory}MB',
                    'suggested_type': f'{suggested_memory}MB',
                    'current_cost': current_cost,
                    'suggested_cost': suggested_cost,
                    'monthly_savings': round(savings, 2),
                    'savings_percentage': round(savings_pct, 1)
                }
        return None
    
    def _get_elasticache_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        current_type = current_config.get('node_type', '')
        current_cost = self.elasticache_pricing.get(current_type, 0)
        
        alternatives = {
            'cache.m5.2xlarge': 'cache.m5.xlarge',
            'cache.m5.xlarge': 'cache.m5.large',
            'cache.r5.2xlarge': 'cache.r5.xlarge',
            'cache.r5.xlarge': 'cache.r5.large',
            'cache.t4g.medium': 'cache.t4g.small',
            'cache.t4g.small': 'cache.t4g.micro',
            'cache.t3.medium': 'cache.t3.small',
            'cache.t3.small': 'cache.t3.micro',
        }
        
        if current_type in alternatives:
            suggested_type = alternatives[current_type]
            suggested_cost = self.elasticache_pricing.get(suggested_type, 0)
            
            if suggested_cost < current_cost:
                savings = current_cost - suggested_cost
                savings_pct = (savings / current_cost) * 100
                
                return {
                    'current_type': current_type,
                    'suggested_type': suggested_type,
                    'current_cost': current_cost,
                    'suggested_cost': suggested_cost,
                    'monthly_savings': round(savings, 2),
                    'savings_percentage': round(savings_pct, 1)
                }
        return None
    
    def _get_ebs_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        current_type = current_config.get('type', 'gp3')
        current_size = current_config.get('size', 20)
        
        # Suggest gp3 over gp2 for cost savings
        if current_type == 'gp2':
            current_cost = self._calculate_ebs_cost(current_config)
            
            suggested_config = current_config.copy()
            suggested_config['type'] = 'gp3'
            suggested_cost = self._calculate_ebs_cost(suggested_config)
            
            savings = current_cost - suggested_cost
            if savings > 0:
                savings_pct = (savings / current_cost) * 100
                
                return {
                    'current_type': f'{current_type} {current_size}GB',
                    'suggested_type': f'gp3 {current_size}GB',
                    'current_cost': current_cost,
                    'suggested_cost': suggested_cost,
                    'monthly_savings': round(savings, 2),
                    'savings_percentage': round(savings_pct, 1)
                }
        return None
    
    def _get_dynamodb_rightsizing(self, current_config: Dict) -> Optional[Dict]:
        billing_mode = current_config.get('billing_mode', 'PAY_PER_REQUEST')
        
        # Suggest switching from provisioned to on-demand for low usage
        if billing_mode == 'PROVISIONED':
            read_capacity = current_config.get('read_capacity', 5)
            write_capacity = current_config.get('write_capacity', 5)
            
            # If provisioned capacity is low, on-demand might be cheaper
            if read_capacity <= 5 and write_capacity <= 5:
                current_cost = self._calculate_dynamodb_cost(current_config)
                
                suggested_config = current_config.copy()
                suggested_config['billing_mode'] = 'PAY_PER_REQUEST'
                suggested_config['estimated_read_units_per_month'] = read_capacity * 730 * 3600  # Assume 1 RRU per second
                suggested_config['estimated_write_units_per_month'] = write_capacity * 730 * 3600  # Assume 1 WRU per second
                suggested_cost = self._calculate_dynamodb_cost(suggested_config)
                
                if suggested_cost < current_cost:
                    savings = current_cost - suggested_cost
                    savings_pct = (savings / current_cost) * 100
                    
                    return {
                        'current_type': 'Provisioned capacity',
                        'suggested_type': 'On-demand',
                        'current_cost': current_cost,
                        'suggested_cost': suggested_cost,
                        'monthly_savings': round(savings, 2),
                        'savings_percentage': round(savings_pct, 1)
                    }
        return None
    
    def _safe_int(self, value: Union[str, int, float], default: int = 0) -> int:
        """Safely convert a value to integer"""
        try:
            if isinstance(value, str):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value: Union[str, int, float], default: float = 0.0) -> float:
        """Safely convert a value to float"""
        try:
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def _calculate_lambda_cost(self, config: Dict) -> float:
        """Calculate Lambda function cost based on memory, duration, and requests"""
        memory_mb = self._safe_int(config.get('memory_size', 128), 128)
        timeout_seconds = self._safe_int(config.get('timeout', 3), 3)
        estimated_requests_per_month = self._safe_int(config.get('estimated_requests', 100000), 100000)
        
        # Convert memory to GB
        memory_gb = memory_mb / 1024
        
        # Calculate GB-seconds per month
        gb_seconds_per_month = memory_gb * timeout_seconds * estimated_requests_per_month
        
        # Apply free tier
        billable_gb_seconds = max(0, gb_seconds_per_month - self.lambda_pricing['free_tier_gb_seconds'])
        billable_requests = max(0, estimated_requests_per_month - self.lambda_pricing['free_tier_requests'])
        
        # Calculate costs
        compute_cost = billable_gb_seconds * self.lambda_pricing['gb_seconds']
        request_cost = (billable_requests / 1000000) * self.lambda_pricing['requests_per_million']
        
        return compute_cost + request_cost
    
    def _calculate_api_gateway_cost(self, config: Dict, api_type: str) -> float:
        """Calculate API Gateway cost based on requests"""
        estimated_requests_per_month = self._safe_int(config.get('estimated_requests', 1000000), 1000000)
        
        if api_type == 'rest':
            cost_per_million = self.api_gateway_pricing['rest_api_requests']
        else:  # HTTP API
            cost_per_million = self.api_gateway_pricing['http_api_requests']
        
        return (estimated_requests_per_month / 1000000) * cost_per_million
    
    def _calculate_cloudfront_cost(self, config: Dict) -> float:
        """Calculate CloudFront distribution cost"""
        estimated_gb_per_month = self._safe_int(config.get('estimated_data_transfer_gb', 100), 100)
        estimated_requests_per_month = self._safe_int(config.get('estimated_requests', 1000000), 1000000)
        
        data_transfer_cost = estimated_gb_per_month * self.cloudfront_pricing['data_transfer_out_gb']
        request_cost = (estimated_requests_per_month / 10000) * self.cloudfront_pricing['https_requests_per_10k']
        
        return data_transfer_cost + request_cost
    
    def _calculate_load_balancer_cost(self, config: Dict) -> float:
        """Calculate Load Balancer cost"""
        lb_type = config.get('load_balancer_type', 'application')
        estimated_lcu_hours = self._safe_int(config.get('estimated_lcu_hours', 1), 1)
        
        if lb_type == 'application':
            base_cost = self.elb_pricing['application_lb_hour']
            lcu_cost = estimated_lcu_hours * self.elb_pricing['alb_lcu_hour']
        elif lb_type == 'network':
            base_cost = self.elb_pricing['network_lb_hour']
            lcu_cost = estimated_lcu_hours * self.elb_pricing['nlb_lcu_hour']
        elif lb_type == 'gateway':
            base_cost = self.elb_pricing['gateway_lb_hour']
            lcu_cost = estimated_lcu_hours * self.elb_pricing['gwlb_lcu_hour']
        else:  # classic
            base_cost = self.elb_pricing['classic_lb_hour']
            lcu_cost = 0
        
        return base_cost + lcu_cost
    
    def _calculate_ebs_cost(self, config: Dict) -> float:
        """Calculate EBS volume cost"""
        size_gb = self._safe_int(config.get('size', 20), 20)
        volume_type = config.get('type', 'gp3')
        iops = self._safe_int(config.get('iops', 3000), 3000)
        
        base_cost = size_gb * self.storage_pricing.get(volume_type, 0.08)
        
        # Add IOPS cost for io1/io2
        if volume_type in ['io1', 'io2'] and iops > size_gb * 3:
            additional_iops = iops - (size_gb * 3)
            iops_cost = additional_iops * 0.065  # $0.065 per provisioned IOPS per month
            base_cost += iops_cost
        
        return base_cost
    
    def _calculate_elastic_ip_cost(self, config: Dict) -> float:
        """Calculate Elastic IP cost"""
        # Assume unused if not explicitly attached
        is_attached = config.get('instance', None) is not None
        return 0.0 if is_attached else self.elastic_ip_pricing['unused_ip']
    
    def _calculate_nat_gateway_cost(self, config: Dict) -> float:
        """Calculate NAT Gateway cost"""
        estimated_data_gb = self._safe_int(config.get('estimated_data_processing_gb', 100), 100)
        
        base_cost = self.nat_gateway_pricing['hourly']
        data_cost = estimated_data_gb * self.nat_gateway_pricing['data_processing_gb']
        
        return base_cost + data_cost
    
    def _calculate_ecs_service_cost(self, config: Dict) -> float:
        """Calculate ECS service cost (depends on launch type)"""
        launch_type = config.get('launch_type', 'EC2')
        
        if launch_type.upper() == 'FARGATE':
            # For ECS service, we need to estimate Fargate costs
            # Since we don't have the actual task definition config here,
            # we'll use default values and multiply by desired count
            desired_count = self._safe_int(config.get('desired_count', 1), 1)
            
            # Default Fargate task configuration
            default_fargate_config = {
                'cpu': 256,  # 0.25 vCPU
                'memory': 512,  # 512 MB
                'estimated_task_hours_per_month': 730  # Full month
            }
            
            single_task_cost = self._calculate_fargate_cost(default_fargate_config)
            return single_task_cost * desired_count
        else:
            # EC2 launch type - cost is in the EC2 instances
            return 0.0
    
    def _calculate_fargate_cost(self, config: Dict) -> float:
        """Calculate Fargate task cost"""
        # Handle case where config might be a string reference
        if not isinstance(config, dict):
            # If it's not a dict, use default values
            config = {
                'cpu': 256,
                'memory': 512,
                'estimated_task_hours_per_month': 730
            }
        
        cpu_units = self._safe_int(config.get('cpu', 256), 256)  # CPU units (256 = 0.25 vCPU)
        memory_mb = self._safe_int(config.get('memory', 512), 512)  # Memory in MB
        estimated_task_hours = self._safe_int(config.get('estimated_task_hours_per_month', 730), 730)  # Default full month
        
        # Convert to vCPU and GB
        vcpu = cpu_units / 1024
        memory_gb = memory_mb / 1024
        
        vcpu_cost = vcpu * estimated_task_hours * (self.container_pricing['fargate_vcpu_hour'] / 730)
        memory_cost = memory_gb * estimated_task_hours * (self.container_pricing['fargate_memory_gb_hour'] / 730)
        
        return vcpu_cost + memory_cost
    
    def _calculate_dynamodb_cost(self, config: Dict) -> float:
        """Calculate DynamoDB table cost"""
        billing_mode = config.get('billing_mode', 'PAY_PER_REQUEST')
        
        if billing_mode == 'PAY_PER_REQUEST':
            # On-demand pricing
            estimated_read_units = self._safe_int(config.get('estimated_read_units_per_month', 1000000), 1000000)
            estimated_write_units = self._safe_int(config.get('estimated_write_units_per_month', 1000000), 1000000)
            
            read_cost = (estimated_read_units / 1000000) * self.dynamodb_pricing['on_demand_read_units']
            write_cost = (estimated_write_units / 1000000) * self.dynamodb_pricing['on_demand_write_units']
            
            return read_cost + write_cost
        else:
            # Provisioned capacity
            read_capacity = self._safe_int(config.get('read_capacity', 5), 5)
            write_capacity = self._safe_int(config.get('write_capacity', 5), 5)
            
            read_cost = read_capacity * self.dynamodb_pricing['provisioned_read_units']
            write_cost = write_capacity * self.dynamodb_pricing['provisioned_write_units']
            
            return read_cost + write_cost
    
    def _calculate_elasticache_cost(self, config: Dict) -> float:
        """Calculate ElastiCache cluster cost"""
        node_type = config.get('node_type', 'cache.t3.micro')
        num_cache_nodes = self._safe_int(config.get('num_cache_nodes', 1), 1)
        
        node_cost = self.elasticache_pricing.get(node_type, 11.68)
        return node_cost * num_cache_nodes
    
    def _calculate_elasticache_replication_cost(self, config: Dict) -> float:
        """Calculate ElastiCache replication group cost"""
        node_type = config.get('node_type', 'cache.t3.micro')
        num_cache_clusters = self._safe_int(config.get('num_cache_clusters', 2), 2)
        replicas_per_node_group = self._safe_int(config.get('replicas_per_node_group', 1), 1)
        
        total_nodes = num_cache_clusters * (1 + replicas_per_node_group)  # Primary + replicas
        node_cost = self.elasticache_pricing.get(node_type, 11.68)
        
        return node_cost * total_nodes