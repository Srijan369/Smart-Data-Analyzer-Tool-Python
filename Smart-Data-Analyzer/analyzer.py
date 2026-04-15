# analyzer.py - Data analysis module with memory-efficient processing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Optimized data analysis class with memory-efficient operations
    for large datasets (40MB+)
    """
    
    def __init__(self, df, sample_size=50000):
        self.df = df
        self.sample_size = sample_size
        self._sample_df = None
        self._numeric_cols = None
        self._categorical_cols = None
        self._datetime_cols = None
        
    def _get_sample(self):
        """
        Get a sample of the data for analysis (critical for large datasets)
        """
        if self._sample_df is not None:
            return self._sample_df
            
        if len(self.df) > self.sample_size:
            self._sample_df = self.df.sample(n=self.sample_size, random_state=42)
        else:
            self._sample_df = self.df
            
        return self._sample_df
        
    def _get_column_types(self):
        """
        Cache column types for efficiency
        """
        if self._numeric_cols is None:
            self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self._categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            self._datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
            
        return self._numeric_cols, self._categorical_cols, self._datetime_cols

    def kpis(self):
        """
        Generate key performance indicators efficiently
        Uses sampling for large datasets
        """
        sample_df = self._get_sample()
        numeric_cols, _, datetime_cols = self._get_column_types()
        
        kpis = {
            "Total Records": f"{len(self.df):,}",
            "Total Columns": len(self.df.columns),
            "Memory Usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        }
        
        # Add numeric KPIs (using sampling for large datasets)
        for col in numeric_cols[:3]:
            if col in sample_df.columns:
                kpis[f"{col} Mean"] = round(float(sample_df[col].mean()), 2)
                kpis[f"{col} Std"] = round(float(sample_df[col].std()), 2)
                
        # Add date range if available
        for col in datetime_cols[:1]:
            if col in self.df.columns:
                kpis["Date Start"] = str(self.df[col].min())[:10]
                kpis["Date End"] = str(self.df[col].max())[:10]
                
        return kpis

    def insights(self, max_insights=8):
        """
        Generate automated insights using sampling for performance
        """
        msgs = []
        sample_df = self._get_sample()
        numeric_cols, categorical_cols, datetime_cols = self._get_column_types()
        
        # 1. Trend analysis (if date and numeric columns exist)
        if datetime_cols and numeric_cols:
            try:
                for nc in numeric_cols[:2]:
                    if nc in sample_df.columns:
                        # Use sample for trend analysis
                        trend_df = sample_df[[datetime_cols[0], nc]].dropna()
                        if len(trend_df) >= 10:
                            trend_df = trend_df.sort_values(datetime_cols[0])
                            if len(trend_df) > 100:
                                # Resample for large time series
                                trend_df = trend_df.set_index(datetime_cols[0])
                                trend_df = trend_df.resample('D').mean().reset_index()
                            
                            if len(trend_df) >= 2:
                                t = trend_df[nc].pct_change().mean() * 100
                                if not pd.isna(t):
                                    if t > 5:
                                        msgs.append(f"📈 **{nc}** has a strong upward trend (+{t:.1f}%).")
                                    elif t > 0:
                                        msgs.append(f"📈 **{nc}** is slightly trending up (+{t:.1f}%).")
                                    elif t < -5:
                                        msgs.append(f"📉 **{nc}** has a strong downward trend ({t:.1f}%).")
                                    elif t < 0:
                                        msgs.append(f"📉 **{nc}** is slightly trending down ({t:.1f}%).")
            except Exception as e:
                pass
                
        # 2. Correlation analysis (using sampling)
        if len(numeric_cols) >= 2:
            try:
                # Use sample for correlation
                corr_df = sample_df[numeric_cols].dropna()
                if len(corr_df) > 1000:
                    corr_df = corr_df.sample(n=min(1000, len(corr_df)), random_state=42)
                    
                if len(corr_df) >= 10:
                    corr = corr_df.corr()
                    cols = corr.columns
                    for i in range(len(cols)):
                        for j in range(i+1, len(cols)):
                            val = corr.iloc[i, j]
                            if abs(val) >= 0.7:
                                d = "positive" if val > 0 else "negative"
                                msgs.append(f"🔗 Strong {d} correlation ({val:.2f}) between **{cols[i]}** and **{cols[j]}**.")
                                if len(msgs) >= max_insights - 2:
                                    break
                        if len(msgs) >= max_insights - 2:
                            break
            except Exception:
                pass
                
        # 3. Data quality insights
        missing_total = int(self.df.isnull().sum().sum())
        if missing_total > 0:
            msgs.append(f"⚠️ {missing_total:,} missing value(s) remain. Use cleaning to fix.")
        else:
            msgs.append("✅ No missing values — excellent data quality!")
            
        # 4. Categorical insights
        if categorical_cols:
            try:
                for cat_col in categorical_cols[:2]:
                    if cat_col in sample_df.columns:
                        top_values = sample_df[cat_col].value_counts().head(3)
                        if len(top_values) > 0:
                            msgs.append(f"📊 **{cat_col}** top values: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}")
            except Exception:
                pass
                
        # 5. Cardinality insights
        if categorical_cols:
            high_cardinality = []
            for col in categorical_cols[:3]:
                if col in sample_df.columns:
                    unique_count = sample_df[col].nunique()
                    if unique_count > 100:
                        high_cardinality.append(f"{col} ({unique_count})")
            if high_cardinality:
                msgs.append(f"⚠️ High cardinality columns: {', '.join(high_cardinality[:2])}")
                
        return msgs[:max_insights] if msgs else ["💡 No significant patterns detected yet."]
        
    def get_statistics(self):
        """
        Get statistical summary efficiently
        """
        numeric_cols, _, _ = self._get_column_types()
        
        if not numeric_cols:
            return None
            
        # Use sample for large datasets
        sample_df = self._get_sample()
        stats_df = sample_df[numeric_cols].describe()
        
        return stats_df.round(2)
        
    def get_distribution(self, column, bins=30):
        """
        Get distribution data for a column (memory efficient)
        """
        if column not in self.df.columns:
            return None
            
        sample_df = self._get_sample()
        
        if column in sample_df.select_dtypes(include=[np.number]).columns:
            hist, bin_edges = np.histogram(sample_df[column].dropna(), bins=bins)
            return {
                'type': 'numeric',
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'min': float(sample_df[column].min()),
                'max': float(sample_df[column].max()),
                'mean': float(sample_df[column].mean()),
                'median': float(sample_df[column].median())
            }
        else:
            value_counts = sample_df[column].value_counts().head(20)
            return {
                'type': 'categorical',
                'values': value_counts.index.tolist(),
                'counts': value_counts.values.tolist(),
                'unique_count': int(sample_df[column].nunique())
            }
            
    def get_correlation_matrix(self, max_cols=10):
        """
        Get correlation matrix efficiently (limited to max_cols)
        """
        numeric_cols, _, _ = self._get_column_types()
        
        if len(numeric_cols) < 2:
            return None
            
        # Limit columns to avoid memory issues
        if len(numeric_cols) > max_cols:
            # Select columns with highest variance
            sample_df = self._get_sample()
            variances = sample_df[numeric_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(max_cols).index.tolist()
        else:
            selected_cols = numeric_cols
            
        # Use sample for correlation
        sample_df = self._get_sample()
        corr_df = sample_df[selected_cols].dropna()
        
        if len(corr_df) > 10000:
            corr_df = corr_df.sample(n=10000, random_state=42)
            
        return corr_df.corr()
        
    def detect_outliers(self, method='iqr'):
        """
        Detect outliers efficiently using sampling
        """
        numeric_cols, _, _ = self._get_column_types()
        outliers = {}
        
        sample_df = self._get_sample()
        
        for col in numeric_cols:
            if col in sample_df.columns:
                data = sample_df[col].dropna()
                if len(data) > 0:
                    if method == 'iqr':
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        
                        outlier_count = len(data[(data < lower) | (data > upper)])
                        outlier_pct = (outlier_count / len(data)) * 100
                        
                        if outlier_count > 0:
                            outliers[col] = {
                                'count': outlier_count,
                                'percentage': round(outlier_pct, 1),
                                'lower_bound': float(lower),
                                'upper_bound': float(upper),
                                'min': float(data.min()),
                                'max': float(data.max())
                            }
                            
        return outliers
        
    def get_column_info(self):
        """
        Get comprehensive column information
        """
        info = []
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            nulls = int(self.df[col].isnull().sum())
            null_pct = round((nulls / len(self.df)) * 100, 1)
            unique = int(self.df[col].nunique())
            
            # Get sample values (memory efficient)
            sample_vals = self.df[col].dropna().head(3).tolist()
            sample_str = ', '.join([str(v)[:50] for v in sample_vals]) if sample_vals else '—'
            
            info.append({
                'Column': col,
                'Type': dtype,
                'Non-Null': f"{len(self.df) - nulls:,}",
                'Nulls': f"{nulls:,} ({null_pct}%)",
                'Unique': f"{unique:,}",
                'Sample': sample_str
            })
            
        return pd.DataFrame(info)
        
    def get_memory_breakdown(self):
        """
        Get memory usage breakdown by column
        """
        memory_usage = self.df.memory_usage(deep=True)
        total_mb = memory_usage.sum() / 1024**2
        
        breakdown = []
        for col, mem in memory_usage.items():
            if col != 'Index':
                mb = mem / 1024**2
                pct = (mb / total_mb) * 100
                breakdown.append({
                    'Column': col,
                    'Memory (MB)': round(mb, 2),
                    'Percentage': round(pct, 1)
                })
                
        return pd.DataFrame(breakdown).sort_values('Memory (MB)', ascending=False)