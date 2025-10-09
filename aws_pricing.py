from typing import Dict, Optional

class AWSPricing:
    def __init__(self):
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
        }
        
        self.rds_pricing = {
            'db.t3.micro': 14.60,
            'db.t3.small': 29.20,
            'db.t3.medium': 58.40,
            'db.t3.large': 116.80,
            'db.t3.xlarge': 233.60,
            'db.t3.2xlarge': 467.20,
        }
        
        self.storage_pricing = {
            'gp2': 0.10,
            'gp3': 0.08,
            'io1': 0.125,
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
            return 5.0
        
        else:
            return 0.0
    
    def get_rightsizing_recommendation(self, resource_type: str, current_config: Dict) -> Optional[Dict]:
        if resource_type == 'aws_instance':
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
        
        elif resource_type == 'aws_rds_instance':
            current_class = current_config.get('instance_class', '')
            current_cost = self.get_rds_cost(current_class)
            
            alternatives = {
                'db.t3.2xlarge': 'db.t3.xlarge',
                'db.t3.xlarge': 'db.t3.large',
                'db.t3.large': 'db.t3.medium',
                'db.t3.medium': 'db.t3.small',
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