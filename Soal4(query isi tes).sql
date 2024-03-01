-- Test Case 1: Channel Analysis
-- Pendapatan total yang dihasilkan dari setiap kelompok channel untuk 5 negara teratas 

SELECT
    channelGrouping,
    country,
    SUM(productRevenue) AS totalRevenue
FROM
    ecommerce_sessions
WHERE
    country IN (
        SELECT
            country
        FROM
            ecommerce_sessions
        GROUP BY
            country
        ORDER BY
            SUM(productRevenue) DESC
        LIMIT 5
    )
GROUP BY
    channelGrouping, country
ORDER BY
    country, totalRevenue DESC;
	
	
-- Test Case 2: User Behavior Analysis
-- Hitung rata-rata timeOnSite, rata-rata pageviews, dan rata-rata sessionQualityDim untuk setiap fullVisitorId. Identifikasi pengguna yang menghabiskan waktu di situs tersebut

WITH UserBehaviorMetrics AS (
    SELECT
        fullVisitorId,
        AVG(timeOnSite) AS avgTimeOnSite,
        AVG(pageviews) AS avgPageviews,
        AVG(sessionQualityDim) AS avgSessionQualityDim
    FROM
        ecommerce_sessions
    GROUP BY
        fullVisitorId
)

SELECT
    ubm.fullVisitorId,
    ubm.avgTimeOnSite,
    ubm.avgPageviews,
    ubm.avgSessionQualityDim
FROM
    UserBehaviorMetrics ubm
JOIN
    ecommerce_sessions es ON ubm.fullVisitorId = es.fullVisitorId
WHERE
    es.timeOnSite > ubm.avgTimeOnSite AND es.pageviews < ubm.avgPageviews;


-- Test Case 3: Product Performance
-- Hitung total pendapatan, total kuantitas terjual, total jumlah pengembalian, peringkat produk berdasarkan pendapatan bersih, dan tandai produk dengan pengembalian melebihi 10% dari total pendapatan.

WITH ProductPerformance AS (
    SELECT
        v2ProductName,
        SUM(productRevenue) AS totalRevenue,
        SUM(productQuantity) AS totalQuantity,
        SUM(productRefundAmount) AS totalRefundAmount,
        (SUM(productRevenue) - SUM(productRefundAmount)) AS netRevenue
    FROM
        ecommerce_sessions
    GROUP BY
        v2ProductName
)

SELECT
    pp.*,
    CASE WHEN pp.totalRefundAmount > 0.1 * pp.totalRevenue THEN 'Flagged' ELSE 'Not Flagged' END AS refundFlag
FROM
    ProductPerformance pp
ORDER BY
    netRevenue DESC;
