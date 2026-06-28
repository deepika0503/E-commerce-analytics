# E-Commerce Analytics — Olist Dataset
# Produces cleaned CSVs for Power BI + inline charts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────
orders      = pd.read_csv(f"{RAW}/olist_orders_dataset.csv", parse_dates=[
                  "order_purchase_timestamp","order_approved_at",
                  "order_delivered_carrier_date","order_delivered_customer_date",
                  "order_estimated_delivery_date"])
customers   = pd.read_csv(f"{RAW}/olist_customers_dataset.csv")
items       = pd.read_csv(f"{RAW}/olist_order_items_dataset.csv")
payments    = pd.read_csv(f"{RAW}/olist_order_payments_dataset.csv")
reviews     = pd.read_csv(f"{RAW}/olist_order_reviews_dataset.csv")
products    = pd.read_csv(f"{RAW}/olist_products_dataset.csv")
sellers     = pd.read_csv(f"{RAW}/olist_sellers_dataset.csv")
cat_trans   = pd.read_csv(f"{RAW}/product_category_name_translation.csv")

print("✅ Data loaded")
print(f"   Orders: {len(orders):,} | Customers: {len(customers):,} | Items: {len(items):,}")

# ── 2. Master Order Table ──────────────────────────────────────────────────
df = (orders
      .merge(customers[["customer_id","customer_unique_id","customer_state"]], on="customer_id")
      .merge(payments.groupby("order_id")["payment_value"].sum().reset_index(), on="order_id", how="left")
      .merge(items.groupby("order_id").agg(
                 item_count=("order_item_id","count"),
                 product_value=("price","sum")).reset_index(), on="order_id", how="left"))

df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
df["order_year"]  = df["order_purchase_timestamp"].dt.year

# ── 3. KPI Summary ─────────────────────────────────────────────────────────
delivered = df[df["order_status"] == "delivered"]

total_revenue   = delivered["payment_value"].sum()
total_orders    = len(delivered)
unique_customers= delivered["customer_unique_id"].nunique()
aov             = total_revenue / total_orders
conversion_rate = len(delivered) / len(df) * 100

kpi = pd.DataFrame({
    "Metric": ["Total Revenue (BRL)", "Total Orders", "Unique Customers",
                "Average Order Value", "Delivery Rate (%)"],
    "Value":  [round(total_revenue,2), total_orders, unique_customers,
               round(aov,2), round(conversion_rate,2)]
})
kpi.to_csv(f"{OUT}/kpi_summary.csv", index=False)
print("\n📊 KPI Summary")
print(kpi.to_string(index=False))

# ── 4. Funnel Analysis ─────────────────────────────────────────────────────
status_order = ["created","approved","processing","invoiced",
                "shipped","delivered","canceled","unavailable"]
funnel_counts = df["order_status"].value_counts()

funnel = pd.DataFrame({
    "Stage": ["Orders Created","Approved","Invoiced","Shipped","Delivered"],
    "Count": [
        len(df),
        df["order_status"].isin(["approved","processing","invoiced","shipped","delivered"]).sum(),
        df["order_status"].isin(["invoiced","shipped","delivered"]).sum(),
        df["order_status"].isin(["shipped","delivered"]).sum(),
        df["order_status"].eq("delivered").sum()
    ]
})
funnel["Drop_off_%"] = (1 - funnel["Count"] / funnel["Count"].shift(1)) * 100
funnel["Drop_off_%"] = funnel["Drop_off_%"].fillna(0).round(2)
funnel.to_csv(f"{OUT}/funnel_analysis.csv", index=False)
print("\n🔽 Funnel")
print(funnel.to_string(index=False))

# ── 5. Monthly Revenue & Orders ────────────────────────────────────────────
monthly = (delivered.groupby("order_month")
           .agg(revenue=("payment_value","sum"),
                orders=("order_id","count"),
                customers=("customer_unique_id","nunique"))
           .reset_index())
monthly["order_month"] = monthly["order_month"].astype(str)
monthly["aov"] = (monthly["revenue"] / monthly["orders"]).round(2)
monthly.to_csv(f"{OUT}/monthly_trends.csv", index=False)
print(f"\n📅 Monthly trends: {len(monthly)} months")

# ── 6. Cohort Retention ────────────────────────────────────────────────────
cohort_df = delivered[["customer_unique_id","order_month"]].copy()
cohort_df["cohort"] = (cohort_df.groupby("customer_unique_id")["order_month"]
                                .transform("min"))
cohort_df["period_number"] = (cohort_df["order_month"].apply(lambda x: x.ordinal) -
                               cohort_df["cohort"].apply(lambda x: x.ordinal))

cohort_pivot = (cohort_df.groupby(["cohort","period_number"])["customer_unique_id"]
                .nunique()
                .unstack())
cohort_size  = cohort_pivot[0]
retention    = cohort_pivot.divide(cohort_size, axis=0).round(4) * 100

retention.index = retention.index.astype(str)
retention.to_csv(f"{OUT}/cohort_retention.csv")
print(f"\n👥 Cohort matrix: {retention.shape[0]} cohorts x {retention.shape[1]} periods")

# ── 7. Category Performance ────────────────────────────────────────────────
cat_items = (items
             .merge(products[["product_id","product_category_name"]], on="product_id", how="left")
             .merge(cat_trans, on="product_category_name", how="left")
             .merge(orders[["order_id","order_status"]], on="order_id"))

cat_items = cat_items[cat_items["order_status"] == "delivered"]
cat_items["category"] = cat_items["product_category_name_english"].fillna(
                         cat_items["product_category_name"]).fillna("Unknown")

cat_perf = (cat_items.groupby("category")
            .agg(revenue=("price","sum"),
                 orders=("order_id","nunique"),
                 items_sold=("order_item_id","count"))
            .reset_index()
            .sort_values("revenue", ascending=False))
cat_perf["revenue"] = cat_perf["revenue"].round(2)
cat_perf["avg_item_price"] = (cat_perf["revenue"] / cat_perf["items_sold"]).round(2)
cat_perf.to_csv(f"{OUT}/category_performance.csv", index=False)
print(f"\n🛍️  Categories: {len(cat_perf)} | Top: {cat_perf.iloc[0]['category']} "
      f"(BRL {cat_perf.iloc[0]['revenue']:,.0f})")

# ── 8. CLV Estimation ─────────────────────────────────────────────────────
clv_df = (delivered.groupby("customer_unique_id")
          .agg(total_spend=("payment_value","sum"),
               order_count=("order_id","count"),
               first_order=("order_purchase_timestamp","min"),
               last_order=("order_purchase_timestamp","max"))
          .reset_index())
clv_df["avg_order_value"]   = (clv_df["total_spend"] / clv_df["order_count"]).round(2)
clv_df["customer_lifespan_days"] = (clv_df["last_order"] - clv_df["first_order"]).dt.days

# Segment by spend
clv_df["segment"] = pd.cut(clv_df["total_spend"],
                            bins=[0,200,600,2000,1e9],
                            labels=["Low (<200)","Mid (200-600)",
                                    "High (600-2k)","VIP (2k+)"])

clv_summary = (clv_df.groupby("segment", observed=True)
               .agg(customers=("customer_unique_id","count"),
                    avg_clv=("total_spend","mean"),
                    avg_orders=("order_count","mean"))
               .reset_index())
clv_summary["avg_clv"]    = clv_summary["avg_clv"].round(2)
clv_summary["avg_orders"] = clv_summary["avg_orders"].round(2)
clv_summary.to_csv(f"{OUT}/clv_segments.csv", index=False)
print("\n💰 CLV Segments")
print(clv_summary.to_string(index=False))

# ── 9. State-level Revenue (for map visual in Power BI) ───────────────────
state_rev = (delivered.groupby("customer_state")
             .agg(revenue=("payment_value","sum"),
                  orders=("order_id","count"))
             .reset_index()
             .sort_values("revenue", ascending=False))
state_rev["revenue"] = state_rev["revenue"].round(2)
state_rev.to_csv(f"{OUT}/state_revenue.csv", index=False)
print(f"\n🗺️  States: {len(state_rev)}")

# ── 10. Review Score Distribution ─────────────────────────────────────────
review_dist = (reviews["review_score"]
               .value_counts()
               .sort_index()
               .reset_index()
               .rename(columns={"index":"score","review_score":"count"}))
review_dist.columns = ["score","count"]
review_dist.to_csv(f"{OUT}/review_scores.csv", index=False)

print("\n✅ All CSVs exported to data/processed/")
print("\nFiles ready for Power BI:")
for f in sorted(os.listdir(OUT)):
    print(f"  📄 {f}")