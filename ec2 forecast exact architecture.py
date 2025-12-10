cat > ec2_forecast_exact_architecture.py << 'ENDOFFILE'
"""
EC2 Forecasting - EXACT GRU Attention Architecture
Matches the exact layer-by-layer structure from training
"""

import boto3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GRU, Dense, Dropout, 
                                     BatchNormalization, Flatten, 
                                     Activation, RepeatVector, Permute, Multiply,
                                     GlobalAveragePooling1D)
from datetime import datetime, timedelta
import json


print(f"TensorFlow Version: {tf.__version__}")

AWS_REGION = 'us-east-1'
EXCLUDE_INSTANCES = ['GRUMODEL']
IDLE_THRESHOLD = 5.0
OPTIMAL_THRESHOLD = 0.58

# ============================================================
# EXACT ARCHITECTURE FROM TRAINING
# ============================================================

def build_exact_gru_attention():
    """
    Rebuild EXACT architecture matching trained model
    Total Parameters: 24,162
    """
    
    
    # INPUT LAYER
    inputs = Input(shape=(12, 3), name='input_layer')
    
    # FIRST GRU LAYER (64 units)
    x = GRU(64, return_sequences=True, name='gru_2')(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Dropout(0.3, name='dropout_6')(x)
    
    # SECOND GRU LAYER (32 units)
    x = GRU(32, return_sequences=True, name='gru_3')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    gru_output = Dropout(0.3, name='dropout_7')(x)  # Save for attention
    
    # ATTENTION MECHANISM (separate layers, not custom class!)
    # Step 1: Compute attention scores
    attention_scores = Dense(1, name='dense')(gru_output)  # (None, 12, 1)
    
    # Step 2: Flatten to 1D
    attention_scores = Flatten(name='flatten')(attention_scores)  # (None, 12)
    
    # Step 3: Softmax to get attention weights
    attention_weights = Activation('softmax', name='activation')(attention_scores)  # (None, 12)
    
    # Step 4: Repeat attention weights 32 times (for 32 GRU units)
    attention_weights = RepeatVector(32, name='repeat_vector')(attention_weights)  # (None, 32, 12)
    
    # Step 5: Permute to match GRU output shape
    attention_weights = Permute([2, 1], name='permute')(attention_weights)  # (None, 12, 32)
    
    # Step 6: Element-wise multiply (apply attention)
    attended = Multiply(name='multiply')([gru_output, attention_weights])  # (None, 12, 32)
    
    # Step 7: Global average pooling (context vector)
    context = GlobalAveragePooling1D(name='global_average_pooling1d')(attended)  # (None, 32)
    
    # CLASSIFICATION HEAD
    x = Dense(32, activation='relu', name='dense_1')(context)
    x = Dropout(0.3, name='dropout_8')(x)
    
    # OUTPUT LAYER
    outputs = Dense(1, activation='sigmoid', name='dense_2')(x)
    
    model = Model(inputs, outputs, name='gru_attention')
    
    
    if model.count_params() != 24162:
        print(f" ")
    else:
        print(f"")
    
    return model

# ============================================================
# DATA COLLECTION
# ============================================================

def collect_ec2_data(instance_id, instance_name):
    print(f"\nCollecting: {instance_name} ({instance_id})")
    
    cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    try:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum', 'Minimum']
        )
        
        if not response['Datapoints']:
            return None
        
        datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
        data = [[p.get('Average', 0), p.get('Maximum', 0), p.get('Minimum', 0)] 
                for p in datapoints]
        data_array = np.array(data)
        
        if len(data_array) < 12:
            print(f"  Only {len(data_array)} datapoints, need 12")
            return None
        
        data_final = data_array[-12:]
        print(f"   Collected 12 timesteps")
        print(f"   Current: Mean={data_final[:, 0].mean():.2f}%, Max={data_final[:, 0].max():.2f}%")
        
        return data_final
        
    except Exception as e:
        print(f"    Error: {e}")
        return None

# ============================================================
# FORECASTING
# ============================================================

def forecast_state(model, cpu_data):
    # Normalize
    cpu_min = cpu_data.min()
    cpu_max = cpu_data.max()
    
    if cpu_max - cpu_min < 1e-7:
        cpu_normalized = np.zeros_like(cpu_data)
    else:
        cpu_normalized = (cpu_data - cpu_min) / (cpu_max - cpu_min)
    
    # Reshape
    X_input = cpu_normalized.reshape(1, 12, 3).astype(np.float32)
    
    # Predict
    y_pred_proba = model.predict(X_input, verbose=0)[0][0]
    
    # Classify
    if y_pred_proba >= OPTIMAL_THRESHOLD:
        forecast_state = "ACTIVE"
        confidence = y_pred_proba * 100
    else:
        forecast_state = "IDLE"
        confidence = (1 - y_pred_proba) * 100
    
    # Current state
    current_max = cpu_data[:, 0].max()
    current_state = "ACTIVE" if current_max >= IDLE_THRESHOLD else "IDLE"
    
    return {
        'current_state': current_state,
        'current_max_cpu': current_max,
        'forecast_state': forecast_state,
        'confidence': confidence,
        'probability': y_pred_proba
    }

def generate_recommendation(result):
    current = result['current_state']
    forecast = result['forecast_state']
    confidence = result['confidence']
    
    if current == forecast:
        if forecast == "ACTIVE":
            action = "CONTINUE MONITORING"
            priority = "MEDIUM"
            reason = f"VM will remain ACTIVE (confidence: {confidence:.1f}%)"
            impact = "Capacity sufficient, continue normal operations"
        else:
            action = "CONSOLIDATION CANDIDATE"
            priority = "HIGH"
            reason = f"VM will remain IDLE (confidence: {confidence:.1f}%)"
            impact = "Schedule consolidation in 60 minutes, save $91/year"
    else:
        if forecast == "ACTIVE":
            action = " PREPARE FOR WORKLOAD"
            priority = "HIGH"
            reason = f"VM will transition IDLE→ACTIVE (confidence: {confidence:.1f}%)"
            impact = "Provision capacity NOW, workload arriving in 60 min"
        else:
            action = " WORKLOAD ENDING"
            priority = "MEDIUM"
            reason = f"VM will transition ACTIVE→IDLE (confidence: {confidence:.1f}%)"
            impact = "Prepare for consolidation in 60 minutes"
    
    return {'action': action, 'priority': priority, 'reason': reason, 'impact': impact}

# ============================================================
# MAIN
# ============================================================

def main():
    
   
    
    model = build_exact_gru_attention()
    
  
    weights_path = 'models/gru_attention_best.h5'
    
    try:
        model.load_weights(weights_path)
        
    except Exception as e:
        
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get EC2 instances
    print(" DISCOVERING EC2 INSTANCES")
    
    ec2 = boto3.client('ec2', region_name=AWS_REGION)
    response = ec2.describe_instances(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )
    
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            name = 'N/A'
            if 'Tags' in instance:
                for tag in instance['Tags']:
                    if tag['Key'] == 'Name':
                        name = tag['Value']
                        break
            
            if name not in EXCLUDE_INSTANCES:
                instances.append({
                    'InstanceId': instance['InstanceId'],
                    'Name': name,
                    'InstanceType': instance.get('InstanceType', 'unknown')
                })
    
    print(f"\nFound {len(instances)} instances to forecast:")
    for i, inst in enumerate(instances, 1):
        print(f"   {i}. {inst['Name']} ({inst['InstanceType']})")
    
    # Forecast each
    results = []
    
    for i, inst in enumerate(instances, 1):
        print(f"\n{'='*70}")
        print(f"  INSTANCE {i}/{len(instances)}: {inst['Name']}")
        print(f"{'='*70}")
        
        cpu_data = collect_ec2_data(inst['InstanceId'], inst['Name'])
        
        if cpu_data is not None:
            result = forecast_state(model, cpu_data)
            
            print(f"\n 60-Minute Forecast:")
            print(f"   Current State: {result['current_state']} (max CPU: {result['current_max_cpu']:.1f}%)")
            print(f"   Forecast (t+60min): {result['forecast_state']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Probability (ACTIVE): {result['probability']:.3f}")
            
            if result['current_state'] != result['forecast_state']:
                print(f"\n    STATE CHANGE PREDICTED!")
                print(f"   {result['current_state']} → {result['forecast_state']} in 60 minutes")
            else:
                print(f"\n   State stable ({result['forecast_state']})")
            
            recommendation = generate_recommendation(result)
            
            print(f"\n Recommendation:")
            print(f"   Action: {recommendation['action']}")
            print(f"   Priority: {recommendation['priority']}")
            print(f"   Reason: {recommendation['reason']}")
            print(f"   Business Impact: {recommendation['impact']}")
            
            results.append({
                'instance_name': inst['Name'],
                'instance_id': inst['InstanceId'],
                'current_state': result['current_state'],
                'current_max_cpu': result['current_max_cpu'],
                'forecast_state': result['forecast_state'],
                'confidence': result['confidence'],
                'state_change': result['current_state'] != result['forecast_state'],
                'recommendation': recommendation,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print(" FORECASTING SUMMARY")
        print(f"{'='*70}")
        
        total = len(results)
        state_changes = sum(1 for r in results if r['state_change'])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n Overall Statistics:")
        print(f"   Total Forecasts: {total}")
        print(f"   State Changes Predicted: {state_changes} ({state_changes/total*100:.1f}%)")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        
        
        print(f"\n{'='*70}")
        print(" FORECAST SUMMARY")
        print(f"{'='*70}")
        
        for r in results:
            change_icon = "" if r['state_change'] else ""
            print(f"\n{change_icon} {r['instance_name']}:")
            print(f"   Current: {r['current_state']} (max CPU: {r['current_max_cpu']:.1f}%)")
            print(f"   Forecast (t+60min): {r['forecast_state']} ({r['confidence']:.1f}% confidence)")
            if r['state_change']:
                print(f"     STATE CHANGE: {r['current_state']} → {r['forecast_state']}")
            print(f"   Action: {r['recommendation']['action']}")
        
        # Save results
        output_file = f"ec2_forecast_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'model': 'GRU Attention (Azure-trained)',
                'architecture': 'Exact match - 24,162 parameters',
                'task': '60-minute ahead forecasting',
                'region': AWS_REGION,
                'total_instances': total,
                'state_changes_predicted': state_changes,
                'average_confidence': float(avg_confidence),
                'results': results
            }, f, indent=2)
        
       
        
        
    
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nForecasting interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

ENDOFFILE

echo "Created: ec2_forecast_exact_architecture.py"
