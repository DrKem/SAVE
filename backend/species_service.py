import pandas as pd
from backend.config import Config

class SpeciesService:
    def __init__(self):
        try:
            # Load all CSV files with proper encoding and error handling
            self.animals_df = pd.read_csv(Config.ANIMAL_DATA)
            self.population_df = pd.read_csv(Config.POPULATION_DATA)
            self.factors_df = pd.read_csv(Config.EFFECTING_FACTOR_DATA)
            self.remediation_df = pd.read_csv(Config.REMEDIATION_DATA)
            print(f"Successfully loaded all CSV files")
            print(f"Animals dataset shape: {self.animals_df.shape}")
            print(f"Population dataset shape: {self.population_df.shape}")
            print(f"Factors dataset shape: {self.factors_df.shape}")
            print(f"Remediation dataset shape: {self.remediation_df.shape}")
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            # Initialize empty DataFrames as fallback
            self.animals_df = pd.DataFrame()
            self.population_df = pd.DataFrame()
            self.factors_df = pd.DataFrame()
            self.remediation_df = pd.DataFrame()
    
    def get_species_info(self, species_name):
        """Get comprehensive species information"""
        try:
            # Check if species exists in the dataset
            matching_animals = self.animals_df[self.animals_df['Animal_Name'].str.lower() == species_name.lower()]
            
            if matching_animals.empty:
                print(f"Species '{species_name}' not found in dataset")
                return {
                    'name': species_name,
                    'status': 'Unknown',
                    'message': 'Species not found in our database'
                }
            
            # Get the first matching animal
            animal_info = matching_animals.iloc[0]
            
            # Get population trend
            population_trend = self.get_population_trend(species_name)
            
            # Get threats
            threats = self.get_threats(species_name)
            
            # Get remediation measures
            measures = self.get_remediation_measures(species_name)
            
            # Create the response object with error handling for missing fields
            basic_info = {
                'name': animal_info.get('Animal_Name', species_name),
                'class': animal_info.get('Class', 'Unknown'),
                'weight': animal_info.get('Weight', 'Unknown'),
                'height': animal_info.get('Height', 'Unknown'),
                'habitats': animal_info.get('Habitats', 'Unknown'),
                'status': animal_info.get('Status', 'Unknown')
            }
            
            # Handle countries field which might be missing or in different format
            if 'Country' in animal_info and animal_info['Country']:
                if isinstance(animal_info['Country'], str):
                    basic_info['countries'] = [c.strip() for c in animal_info['Country'].split(',')]
                else:
                    basic_info['countries'] = [str(animal_info['Country'])]
            else:
                basic_info['countries'] = []
            
            return {
                'basic_info': basic_info,
                'population_trend': population_trend,
                'threats': threats,
                'remediation_measures': measures
            }
        except Exception as e:
            print(f"Error getting species info for '{species_name}': {e}")
            # Return a minimal response with error information
            return {
                'name': species_name,
                'status': 'Error',
                'message': f'Error retrieving species information: {str(e)}'
            }
    
    def get_population_trend(self, species_name):
        try:
            # Case-insensitive match
            species_data = self.population_df[
                self.population_df['Animal'].str.lower() == species_name.lower()
            ].sort_values('Year')
            
            if species_data.empty:
                print(f"No population data found for '{species_name}'")
                return {
                    'years': [],
                    'population': [],
                    'decline_rate': 0,
                    'current_population': 0
                }
            
            # Convert data types explicitly to avoid serialization issues
            years = species_data['Year'].astype(str).tolist()
            populations = species_data['Population'].astype(float).tolist()
            
            # Calculate decline rate safely
            decline_rate = self._calculate_decline_rate(species_data)
            
            # Get current population safely
            try:
                current_population = float(species_data.iloc[-1]['Population'])
            except (IndexError, ValueError):
                current_population = 0
            
            return {
                'years': years,
                'population': populations,
                'decline_rate': decline_rate,
                'current_population': current_population
            }
        except Exception as e:
            print(f"Error getting population trend for '{species_name}': {e}")
            return {
                'years': [],
                'population': [],
                'decline_rate': 0,
                'current_population': 0
            }
    
    def _calculate_decline_rate(self, data):
        try:
            if len(data) < 2:
                return 0
            
            # Convert to float explicitly
            first_count = float(data.iloc[0]['Population'])
            last_count = float(data.iloc[-1]['Population'])
            
            if first_count == 0:
                return 0
                
            decline = ((first_count - last_count) / first_count) * 100
            return round(decline, 2)  # Round to 2 decimal places for readability
        except Exception as e:
            print(f"Error calculating decline rate: {e}")
            return 0
    
    def get_threats(self, species_name):
        try:
            # Case-insensitive match
            threats_data = self.factors_df[
                self.factors_df['Animal_Name'].str.lower() == species_name.lower()
            ]
            
            if threats_data.empty:
                print(f"No threats data found for '{species_name}'")
                return []
            
            # Check if Factor column exists
            if 'Factor' not in threats_data.columns:
                print(f"'Factor' column not found in threats data for '{species_name}'")
                return []
                
            # Get the factors string safely
            factors_str = threats_data.iloc[0].get('Factor', '')
            
            # Handle possible non-string values
            if not isinstance(factors_str, str):
                if pd.isna(factors_str):
                    return []
                factors_str = str(factors_str)
                
            # Split the factors and clean them
            return [threat.strip() for threat in factors_str.split(',') if threat.strip()]
        except Exception as e:
            print(f"Error getting threats for '{species_name}': {e}")
            return []
    
    def get_remediation_measures(self, species_name):
        try:
            # Case-insensitive match
            measures_data = self.remediation_df[
                self.remediation_df['Animal_Name'].str.lower() == species_name.lower()
            ]
            
            if measures_data.empty:
                print(f"No remediation measures found for '{species_name}'")
                return {'measures': [], 'effect': ''}
            
            # Check if required columns exist
            measures = []
            if 'Measure_Taken' in measures_data.columns:
                measures_str = measures_data.iloc[0].get('Measure_Taken', '')
                
                # Handle possible non-string values
                if not isinstance(measures_str, str):
                    if pd.isna(measures_str):
                        measures = []
                    else:
                        measures = [str(measures_str)]
                else:
                    measures = [measure.strip() for measure in measures_str.split(',') if measure.strip()]
            
            # Get effect safely
            effect = ''
            if 'Effect_of_Measures' in measures_data.columns:
                effect_val = measures_data.iloc[0].get('Effect_of_Measures', '')
                if not pd.isna(effect_val):
                    effect = str(effect_val)
            
            return {
                'measures': measures,
                'effect': effect
            }
        except Exception as e:
            print(f"Error getting remediation measures for '{species_name}': {e}")
            return {'measures': [], 'effect': ''}