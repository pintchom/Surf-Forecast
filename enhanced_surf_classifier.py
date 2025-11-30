"""
Enhanced Surf Classification System

More detailed surf condition assessment including wind effects,
wave quality analysis, and multi-level classifications.
"""

import numpy as np
import math

class EnhancedSurfClassifier:
    """
    Advanced surf condition classifier that considers:
    - Wave height and period combinations
    - Wind speed and direction effects
    - Multiple quality levels (Poor, Fair, Good, Excellent)
    - Beginner vs Advanced surfer suitability
    """
    
    def __init__(self, beach_facing_direction=270):
        """
        Initialize classifier.
        
        Args:
            beach_facing_direction: Direction beach faces (degrees)
                                  270 = West-facing (most CA beaches)
        """
        self.beach_facing = beach_facing_direction
        
    def classify_surf_conditions(self, wave_height, wave_period, wind_speed=None, wind_direction=None):
        """
        Comprehensive surf condition classification.
        
        Args:
            wave_height: Significant wave height (meters)
            wave_period: Dominant wave period (seconds)
            wind_speed: Wind speed (m/s), optional
            wind_direction: Wind direction (degrees), optional
            
        Returns:
            Dict with detailed surf assessment
        """
        
        # Base wave assessment
        wave_score = self._assess_wave_quality(wave_height, wave_period)
        
        # Wind effects assessment
        wind_effect = self._assess_wind_effects(wind_speed, wind_direction) if wind_speed and wind_direction else 0
        
        # Overall condition score (0-10 scale)
        overall_score = max(0, min(10, wave_score + wind_effect))
        
        # Classification categories
        condition_level = self._score_to_level(overall_score)
        size_category = self._categorize_wave_size(wave_height)
        quality_category = self._categorize_wave_quality(wave_period)
        wind_category = self._categorize_wind_conditions(wind_speed, wind_direction) if wind_speed and wind_direction else "Unknown"
        
        # Surfability assessment
        surfability = self._assess_surfability(wave_height, wave_period, wind_speed, wind_direction)
        
        return {
            'overall_score': round(overall_score, 1),
            'condition_level': condition_level,
            'surfable': overall_score >= 3.0,
            'size_category': size_category,
            'quality_category': quality_category,
            'wind_category': wind_category,
            'surfability': surfability,
            'wave_assessment': {
                'height_m': round(wave_height, 1),
                'period_s': round(wave_period, 1),
                'wave_power': round(wave_height**2 * wave_period, 1),
                'steepness': round(wave_height / (wave_period**2) * 100, 2)  # Wave steepness indicator
            },
            'recommendations': self._generate_recommendations(overall_score, wave_height, wave_period, wind_speed, wind_direction)
        }
    
    def _assess_wave_quality(self, height, period):
        """
        Assess base wave quality on 0-10 scale.
        
        Considers:
        - Optimal height ranges for different skill levels
        - Period quality (longer = better)
        - Wave power (height² × period)
        """
        
        # Height scoring (bell curve around optimal sizes)
        if height < 0.5:
            height_score = 0  # Too small
        elif height < 1.0:
            height_score = height * 2  # Small but rideable
        elif height <= 2.5:
            height_score = 4 + (height - 1.0) * 2  # Sweet spot
        elif height <= 4.0:
            height_score = 7 - (height - 2.5) * 0.5  # Getting big
        elif height <= 6.0:
            height_score = 6 - (height - 4.0) * 2  # Large, challenging
        else:
            height_score = max(0, 2 - (height - 6.0) * 0.5)  # Too big for most
        
        # Period scoring (longer is generally better)
        if period < 6:
            period_score = 0  # Wind chop
        elif period < 8:
            period_score = 1  # Poor
        elif period < 10:
            period_score = 2.5  # Fair
        elif period < 12:
            period_score = 4  # Decent
        elif period < 15:
            period_score = 5.5  # Good
        elif period <= 20:
            period_score = 6  # Excellent
        else:
            period_score = 5.5  # Very long (rare)
        
        # Wave power bonus (energy content)
        wave_power = height**2 * period
        if wave_power > 50:
            power_bonus = min(1.0, (wave_power - 50) / 100)
        else:
            power_bonus = 0
        
        return height_score + period_score + power_bonus
    
    def _assess_wind_effects(self, wind_speed, wind_direction):
        """
        Assess wind effects on surf quality (-3 to +1 scale).
        
        Args:
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees
        """
        
        if wind_speed is None or wind_direction is None:
            return 0
        
        # Calculate wind direction relative to beach
        relative_wind = self._normalize_angle(wind_direction - self.beach_facing)
        
        # Determine wind effect type
        if -45 <= relative_wind <= 45:
            wind_type = "offshore"  # Wind blowing from land to sea
        elif 135 <= relative_wind or relative_wind <= -135:
            wind_type = "onshore"   # Wind blowing from sea to land
        else:
            wind_type = "sideshore" # Cross wind
        
        # Wind speed effects
        wind_strength = self._categorize_wind_strength(wind_speed)
        
        # Scoring based on wind type and strength
        if wind_type == "offshore":
            if wind_strength == "Light":
                return 1.0    # Perfect - cleans up waves
            elif wind_strength == "Moderate":
                return 0.5    # Good - still helpful
            elif wind_strength == "Strong":
                return -0.5   # Too strong - difficult paddling
            else:  # Very Strong
                return -2.0   # Dangerous conditions
        
        elif wind_type == "sideshore":
            if wind_strength == "Light":
                return 0      # Minimal effect
            elif wind_strength == "Moderate":
                return -0.5   # Some texture
            else:
                return -1.5   # Choppy conditions
        
        else:  # onshore
            if wind_strength == "Light":
                return -0.5   # Slight texture
            elif wind_strength == "Moderate":
                return -1.5   # Choppy
            elif wind_strength == "Strong":
                return -2.5   # Very messy
            else:  # Very Strong
                return -3.0   # Blown out
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180] range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def _categorize_wind_strength(self, wind_speed):
        """Categorize wind strength."""
        if wind_speed < 3:
            return "Light"
        elif wind_speed < 7:
            return "Moderate" 
        elif wind_speed < 12:
            return "Strong"
        else:
            return "Very Strong"
    
    def _score_to_level(self, score):
        """Convert numeric score to descriptive level."""
        if score < 2:
            return "Poor"
        elif score < 4:
            return "Fair" 
        elif score < 6:
            return "Good"
        elif score < 8:
            return "Very Good"
        else:
            return "Excellent"
    
    def _categorize_wave_size(self, height):
        """Categorize wave size."""
        if height < 0.6:
            return "Tiny"
        elif height < 1.0:
            return "Small"
        elif height < 1.5:
            return "Small-Medium"
        elif height < 2.0:
            return "Medium"
        elif height < 2.5:
            return "Medium-Large" 
        elif height < 3.5:
            return "Large"
        elif height < 5.0:
            return "Extra Large"
        else:
            return "XXL"
    
    def _categorize_wave_quality(self, period):
        """Categorize wave quality based on period."""
        if period < 6:
            return "Poor (Wind Chop)"
        elif period < 8:
            return "Fair (Short Period)"
        elif period < 10:
            return "Fair-Good"
        elif period < 12:
            return "Good"
        elif period < 15:
            return "Very Good"
        elif period < 18:
            return "Excellent"
        else:
            return "Epic (Groundswell)"
    
    def _categorize_wind_conditions(self, wind_speed, wind_direction):
        """Categorize overall wind conditions."""
        if wind_speed is None or wind_direction is None:
            return "Unknown"
        
        relative_wind = self._normalize_angle(wind_direction - self.beach_facing)
        wind_strength = self._categorize_wind_strength(wind_speed)
        
        if -45 <= relative_wind <= 45:
            wind_direction_desc = "Offshore"
        elif 135 <= relative_wind or relative_wind <= -135:
            wind_direction_desc = "Onshore"
        else:
            wind_direction_desc = "Sideshore"
        
        return f"{wind_direction_desc} - {wind_strength}"
    
    def _assess_surfability(self, height, period, wind_speed=None, wind_direction=None):
        """Assess surfability for different skill levels."""
        
        base_score = self._assess_wave_quality(height, period)
        wind_effect = self._assess_wind_effects(wind_speed, wind_direction) if wind_speed and wind_direction else 0
        total_score = base_score + wind_effect
        
        # Skill level recommendations
        beginner_suitable = (0.8 <= height <= 1.5) and (period >= 8) and (total_score >= 3)
        intermediate_suitable = (0.6 <= height <= 2.5) and (period >= 7) and (total_score >= 2.5)
        advanced_suitable = (0.5 <= height <= 4.0) and (period >= 6) and (total_score >= 2)
        expert_suitable = height >= 0.5 and period >= 5
        
        # Crowd factor (better conditions = more crowded)
        if total_score >= 7:
            crowd_level = "Very Crowded"
        elif total_score >= 5:
            crowd_level = "Crowded"
        elif total_score >= 3:
            crowd_level = "Moderate"
        else:
            crowd_level = "Empty"
        
        return {
            'beginner_friendly': beginner_suitable,
            'intermediate_suitable': intermediate_suitable, 
            'advanced_suitable': advanced_suitable,
            'expert_only': not intermediate_suitable and advanced_suitable,
            'too_dangerous': height > 5.0 or total_score < 1,
            'expected_crowd': crowd_level
        }
    
    def _generate_recommendations(self, score, height, period, wind_speed=None, wind_direction=None):
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Basic condition assessment
        if score >= 7:
            recommendations.append("🏄‍♂️ Excellent conditions - Go surf now!")
        elif score >= 5:
            recommendations.append("🌊 Good conditions - Worth going out")
        elif score >= 3:
            recommendations.append("🤔 Fair conditions - Consider your options")
        else:
            recommendations.append("😔 Poor conditions - Maybe skip today")
        
        # Size-specific advice
        if height < 0.8:
            recommendations.append("📏 Waves quite small - Good for beginners or longboards")
        elif height > 3.0:
            recommendations.append("⚠️  Large waves - Advanced surfers only")
        
        # Period-specific advice
        if period < 8:
            recommendations.append("💨 Short period waves - Expect choppy conditions")
        elif period > 15:
            recommendations.append("🌊 Long period swell - Powerful, well-spaced waves")
        
        # Wind-specific advice
        if wind_speed and wind_direction:
            relative_wind = self._normalize_angle(wind_direction - self.beach_facing)
            
            if -45 <= relative_wind <= 45 and wind_speed < 5:
                recommendations.append("🍃 Light offshore winds - Waves should be clean")
            elif wind_speed > 10:
                recommendations.append("💨 Strong winds - Expect challenging conditions")
        
        # Timing advice
        if score >= 5:
            recommendations.append("⏰ Get there early - Conditions may deteriorate")
        elif 3 <= score < 5:
            recommendations.append("⏰ Monitor conditions - May improve throughout day")
        
        return recommendations

def update_surf_classification_in_predictor():
    """
    Enhanced classification function to replace the simple binary classification
    in the real-time predictor.
    """
    
    def classify_surf_conditions_enhanced(predictions, wind_data=None):
        """
        Enhanced surf classification with detailed analysis.
        
        Args:
            predictions: Dict with model predictions
            wind_data: Optional wind speed and direction data
        """
        
        classifier = EnhancedSurfClassifier()
        enhanced_conditions = {}
        
        for model_name, preds in predictions.items():
            conditions = {}
            
            for horizon in ['1h', '3h', '6h']:
                wvht = preds[f'WVHT_{horizon}']
                dpd = preds[f'DPD_{horizon}']
                
                # Get wind data if available
                wind_speed = wind_data.get('WSPD') if wind_data else None
                wind_direction = wind_data.get('WDIR') if wind_data else None
                
                # Enhanced classification
                surf_analysis = classifier.classify_surf_conditions(
                    wave_height=wvht,
                    wave_period=dpd, 
                    wind_speed=wind_speed,
                    wind_direction=wind_direction
                )
                
                conditions[horizon] = surf_analysis
            
            enhanced_conditions[model_name] = conditions
        
        return enhanced_conditions
    
    return classify_surf_conditions_enhanced

# Example usage and demonstration
if __name__ == "__main__":
    classifier = EnhancedSurfClassifier()
    
    # Test various conditions
    test_conditions = [
        {"height": 1.5, "period": 12, "wind_speed": 3, "wind_direction": 45, "desc": "Good offshore day"},
        {"height": 2.5, "period": 8, "wind_speed": 15, "wind_direction": 225, "desc": "Big but blown out"},
        {"height": 1.0, "period": 14, "wind_speed": 2, "wind_direction": 90, "desc": "Small but clean"},
        {"height": 0.7, "period": 6, "wind_speed": 8, "wind_direction": 180, "desc": "Small and choppy"},
        {"height": 3.5, "period": 16, "wind_speed": 5, "wind_direction": 60, "desc": "Epic day"},
    ]
    
    print("=" * 80)
    print("ENHANCED SURF CLASSIFICATION EXAMPLES")
    print("=" * 80)
    
    for i, test in enumerate(test_conditions, 1):
        result = classifier.classify_surf_conditions(
            test["height"], test["period"], test["wind_speed"], test["wind_direction"]
        )
        
        print(f"\n{i}. {test['desc'].upper()}")
        print("-" * 40)
        print(f"Waves: {test['height']}m @ {test['period']}s")
        print(f"Wind: {test['wind_speed']} m/s from {test['wind_direction']}°")
        print(f"Score: {result['overall_score']}/10 - {result['condition_level']}")
        print(f"Size: {result['size_category']} | Quality: {result['quality_category']}")
        print(f"Wind: {result['wind_category']}")
        print(f"Surfable: {result['surfable']} | Beginner OK: {result['surfability']['beginner_friendly']}")
        print("Recommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")