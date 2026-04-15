# cleaner.py - Data cleaning module with optimized memory management (FIXED TITLE CASE)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Optimized data cleaning class with memory-efficient operations
    for large datasets (40MB+)
    """
    
    def __init__(self, df):
        self.original_df = df.copy() if df is not None else None
        self.df = df.copy() if df is not None else None
        self.cleaning_log = []
        self._original_memory = None
        self._cleaned_memory = None
        
    def capitalize_text(self, text):
        """
        Convert text to proper title case
        Examples: "CHENNAI" -> "Chennai", "pune" -> "Pune", "MUMBAI" -> "Mumbai"
        """
        if pd.isna(text) or not isinstance(text, str):
            return text
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to title case (capitalize first letter of each word)
        # Using str.title() would make "McDonald's" -> "Mcdonald'S", so we use a better approach
        words = text.split()
        capitalized_words = []
        
        for word in words:
            # Handle words with apostrophes or special characters
            if "'" in word or "-" in word:
                # Split by apostrophe or hyphen and capitalize each part
                parts = []
                if "'" in word:
                    parts = word.split("'")
                    capitalized_parts = [p.capitalize() for p in parts]
                    capitalized_words.append("'").join(capitalized_parts)
                elif "-" in word:
                    parts = word.split("-")
                    capitalized_parts = [p.capitalize() for p in parts]
                    capitalized_words.append("-".join(capitalized_parts))
                else:
                    capitalized_words.append(word.capitalize())
            else:
                # Simple capitalize
                capitalized_words.append(word.capitalize())
        
        return ' '.join(capitalized_words)
    
    def optimize_dtypes(self, df=None):
        """
        Optimize column dtypes to reduce memory usage
        This is crucial for large datasets (40MB+)
        """
        if df is None:
            df = self.df
            
        if df is None:
            return df
            
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Skip datetime columns for min/max optimization
            if 'datetime' in str(col_type):
                continue
                
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(col_type):
                # Check if column is all integers
                if pd.api.types.is_integer_dtype(col_type):
                    try:
                        c_min = df[col].min()
                        c_max = df[col].max()
                        
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                    except Exception:
                        pass
                # Handle float columns
                elif pd.api.types.is_float_dtype(col_type):
                    try:
                        c_min = df[col].min()
                        c_max = df[col].max()
                        
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                    except Exception:
                        pass
            
            # Handle object/category columns - convert to category if beneficial
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5 and df[col].nunique() < 5000:
                        df[col] = df[col].astype('category')
                except Exception:
                    pass
                    
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        self.cleaning_log.append(f"✓ Memory optimization: {start_mem:.1f}MB → {end_mem:.1f}MB ({(1 - end_mem/start_mem)*100:.1f}% reduction)")
        
        return df

    def clean_data(self, optimize_memory=True):
        """
        Comprehensive data cleaning with memory optimization
        """
        if self.df is None:
            return None, self.cleaning_log
            
        df = self.df
        
        # Store original memory usage
        self._original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Get first column name (likely ID column)
        first_col = df.columns[0] if len(df.columns) > 0 else None
        
        # 1. Strip whitespace from string columns and apply title casing
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            try:
                # Strip whitespace first
                df[col] = df[col].astype(str).str.strip()
                self.cleaning_log.append(f"✓ Stripped whitespace in '{col}'")
                
                # Apply title casing for all columns except first column
                if col != first_col:
                    df[col] = df[col].apply(self.capitalize_text)
                    self.cleaning_log.append(f"✓ Applied title case to '{col}'")
                else:
                    self.cleaning_log.append(f"✓ Skipped title casing for ID column '{col}' (kept original format)")
                    
            except Exception as e:
                self.cleaning_log.append(f"⚠️ Could not process '{col}': {str(e)}")
                
        # 2. Handle categorical columns (optimize before processing)
        for col in str_cols:
            try:
                if col in df.columns and df[col].nunique() < len(df) * 0.5 and df[col].nunique() < 5000:
                    df[col] = df[col].astype('category')
                    self.cleaning_log.append(f"✓ Converted '{col}' to category for optimization")
            except Exception:
                pass
                
        # 3. Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            self.cleaning_log.append(f"✓ Removed {removed} duplicate row(s)")
            
        # 4. Handle missing values efficiently
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
                
            try:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    # For ID column, use a placeholder, otherwise 'Unknown'
                    if col == first_col:
                        df[col] = df[col].fillna("MISSING_ID")
                        self.cleaning_log.append(f"✓ Filled {null_count} nulls in ID column '{col}' → 'MISSING_ID'")
                    else:
                        df[col] = df[col].fillna("Unknown")
                        self.cleaning_log.append(f"✓ Filled {null_count} nulls in '{col}' → 'Unknown'")
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    # Use forward fill for datetime columns
                    df[col] = df[col].fillna(method='ffill')
                    self.cleaning_log.append(f"✓ Filled {null_count} nulls in '{col}' → forward fill")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    mv = df[col].median()
                    df[col] = df[col].fillna(mv)
                    self.cleaning_log.append(f"✓ Filled {null_count} nulls in '{col}' → median ({mv:.2f})")
            except Exception as e:
                # If fill fails, leave as is
                pass
                
        # 5. Convert string numbers to numeric
        for col in df.select_dtypes(include=['object', 'string']).columns:
            try:
                # Skip ID column for numeric conversion
                if col == first_col:
                    continue
                    
                # Sample first 1000 rows to check if convertible
                sample = df[col].dropna().head(1000)
                if len(sample) > 0:
                    converted = pd.to_numeric(sample, errors='coerce')
                    if converted.notna().sum() > len(sample) * 0.6:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.cleaning_log.append(f"✓ Converted '{col}' to numeric")
            except Exception:
                pass
                
        # 6. Convert to datetime if applicable
        for col in df.select_dtypes(include=['object', 'string']).columns:
            try:
                # Skip ID column for datetime conversion
                if col == first_col:
                    continue
                    
                sample = df[col].dropna().head(1000)
                if len(sample) > 0:
                    converted = pd.to_datetime(sample, errors='coerce')
                    if converted.notna().sum() > len(sample) * 0.6:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        self.cleaning_log.append(f"✓ Converted '{col}' to datetime")
            except Exception:
                pass
                
        # 7. Optimize memory usage (critical for large datasets)
        if optimize_memory:
            df = self.optimize_dtypes(df)
            
        self.df = df
        self._cleaned_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        return self.df, self.cleaning_log

    def quality_score(self, df=None):
        """
        Calculate data quality score
        """
        if df is None:
            df = self.df
            
        if df is None or len(df) == 0 or len(df.columns) == 0:
            return 100.0
            
        tc = df.shape[0] * df.shape[1]
        if tc == 0:
            return 100.0
            
        mp = df.isnull().sum().sum() / tc * 100
        dp = df.duplicated().sum() / len(df) * 100 if len(df) else 0
        
        # Memory efficiency factor
        mem_efficiency = 1.0
        if self._original_memory and self._cleaned_memory:
            mem_efficiency = min(1.0, self._original_memory / self._cleaned_memory) if self._cleaned_memory > 0 else 1.0
            
        score = max(0.0, min(100.0, 100 - mp * 0.5 - dp * 0.3))
        score = score * (0.9 + 0.1 * mem_efficiency)
        
        return round(score, 1)
        
    def get_memory_info(self):
        """
        Get memory usage information
        """
        if self.df is None:
            return {"original": 0, "cleaned": 0, "reduction": 0}
            
        current_mem = self.df.memory_usage(deep=True).sum() / 1024**2
        
        return {
            "original_mb": round(self._original_memory or current_mem, 2),
            "cleaned_mb": round(current_mem, 2),
            "reduction_percent": round(((self._original_memory or current_mem) - current_mem) / (self._original_memory or 1) * 100, 1)
        }
        
    def sample_data(self, n=1000):
        """
        Get a sample of the data for analysis (memory efficient)
        """
        if self.df is None:
            return None
            
        if len(self.df) > n:
            return self.df.sample(n=min(n, len(self.df)), random_state=42)
        return self.df