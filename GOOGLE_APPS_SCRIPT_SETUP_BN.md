# ঝাদিমাদি ডটকম (Jhadimadi.com) - গুগল অ্যাপস স্ক্রিপ্ট অর্ডার ম্যানেজমেন্ট কোড ও নির্দেশিকা

এই স্ক্রিপ্টটি আপনার গুগল স্প্রেডশিটে (`Jhadimadi orders`) একক বা একাধিক (২, ৩ বা ততোধিক পণ্যের কার্ট) অর্ডার স্বয়ংক্রিয়ভাবে সঠিকভাবে যুক্ত করার জন্য তৈরি করা হয়েছে।

---

## 📋 গুগল শিটের কলাম বিন্যাস (Columns A to J):
- **Column A:** `order ID`
- **Column B:** `Date & Time`
- **Column C:** `Customer Name`
- **Column D:** `Phone Number`
- **Column E:** `Delivery Address`
- **Column F:** `Product ID`
- **Column G:** `Product Name`
- **Column H:** `Quality/Size`
- **Column I:** `Total Amount`
- **Column J:** `Status`

---

## 💻 সম্পূর্ণ গুগল অ্যাপস স্ক্রিপ্ট কোড (Code.gs)

```javascript
/**
 * ============================================================================
 * Jhadimadi.com - Google Apps Script Order Management Webhook (doPost & doGet)
 * ============================================================================
 */

var CONFIG = {
  DEFAULT_SHEET_NAME: "orders",
  STOCK_SHEET_NAME: "Stock",
  TIMEZONE: "Asia/Dhaka",
  DATE_FORMAT: "dd/MM/yyyy, hh:mm:ss a"
};

/**
 * HTTP POST রিকোয়েস্ট হ্যান্ডলার - নতুন অর্ডার গ্রহণ ও শিটে রো যুক্ত করা
 */
function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    // একই সাথে একাধিক অর্ডার আসলে ডাটা সুরক্ষিত রাখতে ১০ সেকেন্ড লক অপেক্ষা করা হবে
    lock.waitLock(10000);
  } catch (lockErr) {
    console.warn("LockService notice: " + lockErr.toString());
  }

  try {
    var rawPayload = "";
    var data = null;

    // ১. পেলোড পার্সিং ও এরর হ্যান্ডলিং (try-catch)
    if (e && e.postData && e.postData.contents) {
      rawPayload = e.postData.contents;
      Logger.log("doPost triggered with payload: " + rawPayload);
      try {
        data = JSON.parse(rawPayload);
        // ডাবল-স্ট্রিংগিফাই হ্যান্ডলিং
        if (typeof data === "string") {
          try {
            data = JSON.parse(data);
          } catch (innerErr) {
            Logger.log("Inner parse warning: " + innerErr.toString());
          }
        }
      } catch (parseErr) {
        Logger.log("JSON parse error: " + parseErr.toString() + " | Raw: " + rawPayload);
        console.warn("JSON parse warning: " + parseErr.toString());
      }
    }

    // অল্টারনেটিভ প্যারামিটার হ্যান্ডলিং (URL-encoded fallback)
    if (!data && e && e.parameter) {
      data = e.parameter;
    }

    if (!data) {
      return createJsonResponse({
        success: false,
        message: "কোন ডাটা পাওয়া যায়নি (Empty or unparseable payload)."
      });
    }

    // ২. অ্যাক্টিভ স্প্রেডশিট ও সঠিক শিট ট্যাব নির্বাচন
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = getOrdersSheet(ss);

    if (!sheet) {
      return createJsonResponse({
        success: false,
        message: "গুগল শিটে অর্ডারের কোনো শিট ট্যাব ('orders') খুঁজে পাওয়া যায়নি।"
      });
    }

    // হেডার নিশ্চিত করা
    ensureHeaders(sheet);

    // বর্তমান সময় (যদি পেলোডে না থাকে)
    var currentDateTime = Utilities.formatDate(new Date(), CONFIG.TIMEZONE, CONFIG.DATE_FORMAT);

    // ৩. কমন ফিল্ডসমূহ এক্সট্রাক্ট করা (Shared Order Data)
    var orderId = sanitizeString(data["order ID"] || data.orderId || data.order_id || data.id || ("JDM-ORD-" + Math.floor(100000 + Math.random() * 900000)));
    var dateTime = sanitizeString(data["Date & Time"] || data.dateTime || data.date || currentDateTime);
    var customerName = sanitizeString(data["Customer Name"] || data.customerName || data.name || "Customer");
    
    // ফোন নাম্বার যেন গুগল শিটে সামনের '0' না কাটে, সেজন্য টেক্সট প্রিফিক্স যোগ করা
    var rawPhone = sanitizeString(data["Phone Number"] || data.phoneNumber || data.phone || data.customerPhone || "");
    var phoneNumber = formatPhoneNumber(rawPhone);

    var deliveryAddress = sanitizeString(data["Delivery Address"] || data.deliveryAddress || data.address || "প্রযোজ্য নয়");
    var totalOrderAmount = data["Total Amount"] !== undefined ? data["Total Amount"] : (data.totalAmount || data.totalPrice || 0);
    var status = sanitizeString(data["Status"] || data.status || "Pending");

    // ৩.১ ডুপ্লিকেশন রোধ (Deduplication Check in Sheet)
    if (orderId) {
      var lastRow = sheet.getLastRow();
      if (lastRow > 1) {
        var checkRows = Math.min(50, lastRow - 1);
        var startRow = Math.max(2, lastRow - checkRows + 1);
        var existingOrderIds = sheet.getRange(startRow, 1, checkRows, 1).getValues();
        for (var i = 0; i < existingOrderIds.length; i++) {
          if (String(existingOrderIds[i][0]).trim() === orderId) {
            Logger.log("Order " + orderId + " already recorded. Skipping duplicate.");
            return createJsonResponse({
              result: "success",
              success: true,
              message: "Order already recorded in Google Sheets (deduplicated)",
              orderId: orderId,
              deduplicated: true
            });
          }
        }
      }
    }

    var rowsAdded = 0;

    // ৪. পেলোড বিশ্লেষণ: Array নাকি Single Item?
    // ক) যদি ডাটা নিজেই একটি অ্যারে হয়: [ { ... }, { ... } ]
    if (Array.isArray(data)) {
      data.forEach(function(item) {
        var rowOrderId = sanitizeString(item["order ID"] || item.orderId || orderId);
        var rowDateTime = sanitizeString(item["Date & Time"] || item.dateTime || dateTime);
        var rowCustName = sanitizeString(item["Customer Name"] || item.customerName || customerName);
        var rowPhone = formatPhoneNumber(sanitizeString(item["Phone Number"] || item.phoneNumber || rawPhone));
        var rowAddress = sanitizeString(item["Delivery Address"] || item.deliveryAddress || deliveryAddress);

        var rowProdId = sanitizeString(item["Product ID"] || item.productId || item.code || item.id || "JDM-001");
        var rowProdName = sanitizeString(item["Product Name"] || item.productName || item.name || "পণ্য");
        var rowQualitySize = sanitizeString(item["Quality/Size"] || item.qualitySize || item.formattedQuantity || item.size || (item.quantity ? item.quantity + " টি" : "১ টি"));
        var rowAmount = item["Total Amount"] !== undefined ? item["Total Amount"] : (item.totalAmount || item.price || totalOrderAmount);
        var rowStatus = sanitizeString(item["Status"] || item.status || status);

        appendOrderRow(sheet, [
          rowOrderId,
          rowDateTime,
          rowCustName,
          rowPhone,
          rowAddress,
          rowProdId,
          rowProdName,
          rowQualitySize,
          rowAmount,
          rowStatus
        ]);
        rowsAdded++;
      });
    }
    // খ) যদি পেলোডে items / cartItems / products অ্যারে থাকে:
    else if (
      (Array.isArray(data.items) && data.items.length > 0) ||
      (Array.isArray(data.cartItems) && data.cartItems.length > 0) ||
      (Array.isArray(data.products) && data.products.length > 0)
    ) {
      var itemsList = data.items || data.cartItems || data.products;

      itemsList.forEach(function(item) {
        var itemProdId = sanitizeString(item["Product ID"] || item.productId || item.productCode || item.code || item.id || data.productId || "JDM-001");
        var itemProdName = sanitizeString(item["Product Name"] || item.productName || item.name || item.nameBn || item.title || data.productName || "পণ্য");
        var itemQualitySize = sanitizeString(item["Quality/Size"] || item.qualitySize || item.formattedQuantity || item.size || (item.quantity ? item.quantity + " টি" : data.qualitySize || "১ টি"));
        
        var itemAmount = (item.totalAmount !== undefined && item.totalAmount !== null && item.totalAmount !== "")
          ? item.totalAmount
          : (item.price && item.quantity ? (Number(item.price) * Number(item.quantity)) : totalOrderAmount);
        
        var itemStatus = sanitizeString(item.status || status);

        appendOrderRow(sheet, [
          orderId,            // Column A: order ID
          dateTime,           // Column B: Date & Time
          customerName,       // Column C: Customer Name
          phoneNumber,        // Column D: Phone Number
          deliveryAddress,    // Column E: Delivery Address
          itemProdId,         // Column F: Product ID
          itemProdName,       // Column G: Product Name
          itemQualitySize,    // Column H: Quantity/Size
          itemAmount,         // Column I: Total Amount
          itemStatus          // Column J: Status
        ]);
        rowsAdded++;
      });
    }
    // গ) সেফগার্ড: যদি items অ্যারে না থাকে, কিন্তু একাধিক পণ্যের নাম বা আইডি কমা (,) দিয়ে আলাদা থাকে
    else if (
      (data.productName && String(data.productName).indexOf(",") !== -1) ||
      (data.productId && String(data.productId).indexOf(",") !== -1)
    ) {
      var prodIds = String(data.productId || "JDM-001").split(",").map(function(s) { return s.trim(); });
      var prodNames = String(data.productName || "পণ্য").split(",").map(function(s) { return s.trim(); });
      var qualitySizes = String(data.qualitySize || "১ টি").split(",").map(function(s) { return s.trim(); });

      var maxLen = Math.max(prodIds.length, prodNames.length, qualitySizes.length);

      for (var i = 0; i < maxLen; i++) {
        var singleProdId = prodIds[i] || prodIds[0] || "JDM-001";
        var singleProdName = prodNames[i] || prodNames[0] || "পণ্য";
        var singleQualitySize = qualitySizes[i] || qualitySizes[0] || "১ টি";

        appendOrderRow(sheet, [
          orderId,
          dateTime,
          customerName,
          phoneNumber,
          deliveryAddress,
          singleProdId,
          singleProdName,
          singleQualitySize,
          totalOrderAmount,
          status
        ]);
        rowsAdded++;
      }
    }
    // ঘ) সিঙ্গেল আইটেম অর্ডার (Standard Single-item Order)
    else {
      var singleProductId = sanitizeString(data["Product ID"] || data.productId || data.productCode || data.code || data.id || "JDM-001");
      var singleProductName = sanitizeString(data["Product Name"] || data.productName || data.name || data.nameBn || "পণ্য");
      var singleQualitySize = sanitizeString(data["Quality/Size"] || data.qualitySize || data.formattedQuantity || data.quantity || "১ টি");

      appendOrderRow(sheet, [
        orderId,            // Column A: order ID
        dateTime,           // Column B: Date & Time
        customerName,       // Column C: Customer Name
        phoneNumber,        // Column D: Phone Number
        deliveryAddress,    // Column E: Delivery Address
        singleProductId,    // Column F: Product ID
        singleProductName,  // Column G: Product Name
        singleQualitySize,  // Column H: Quantity/Size
        totalOrderAmount,   // Column I: Total Amount
        status              // Column J: Status
      ]);
      rowsAdded = 1;
    }

    return createJsonResponse({
      result: "success",
      success: true,
      message: "অর্ডার সফলভাবে গুগল শিটে যোগ করা হয়েছে। (" + rowsAdded + " টি আইটেম রো যোগ হয়েছে)",
      orderId: orderId,
      itemsCount: rowsAdded
    });

  } catch (error) {
    Logger.log("doPost critical error: " + error.toString() + (error.stack ? "\n" + error.stack : ""));
    console.error("doPost error: " + error.toString());
    return createJsonResponse({
      result: "error",
      success: false,
      error: error.toString(),
      stack: error.stack ? String(error.stack) : "",
      message: "গুগল শিটে অর্ডার সংরক্ষণে সমস্যা হয়েছে: " + error.toString()
    });
  } finally {
    try {
      lock.releaseLock();
    } catch (_) {}
  }
}

/**
 * HTTP GET রিকোয়েস্ট হ্যান্ডলার - ইনভেন্টরি ও স্টক ডাটা রিটার্ন বা সিস্টেম স্ট্যাটাস চেক
 */
function doGet(e) {
  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var param = (e && e.parameter) ? e.parameter : {};
    var sheetName = param.sheet || param.tab || CONFIG.STOCK_SHEET_NAME;

    // স্টক শিট ফেচিং সাপোর্ট
    if (param.action === "getStock" || sheetName.toLowerCase() === "stock") {
      var stockSheet = ss.getSheetByName(CONFIG.STOCK_SHEET_NAME) || ss.getSheetByName("stock") || ss.getSheetByName("ইনভেন্টরি");
      if (stockSheet) {
        var dataRange = stockSheet.getDataRange().getValues();
        var headers = dataRange[0] || [];
        var rows = dataRange.slice(1);
        var stockList = [];

        for (var i = 0; i < rows.length; i++) {
          var r = rows[i];
          if (!r[0] && !r[1]) continue;
          stockList.push({
            productId: String(r[0] || "").trim(),
            productName: String(r[1] || "").trim(),
            category: String(r[2] || "").trim(),
            currentStock: Number(r[3]) || 0,
            status: String(r[4] || "Available").trim(),
            date: String(r[5] || "").trim()
          });
        }

        return createJsonResponse({
          success: true,
          stock: stockList,
          data: stockList,
          count: stockList.length
        });
      }
    }

    // সাধারণ সিস্টেম স্ট্যাটাস
    return createJsonResponse({
      success: true,
      status: "online",
      message: "Jhadimadi Google Sheet Webhook is active and running.",
      timestamp: new Date().toISOString()
    });

  } catch (err) {
    return createJsonResponse({
      success: false,
      error: err.toString()
    });
  }
}

/**
 * হেল্পার ফাংশন: সঠিক অর্ডারের শিট ট্যাব খুঁজে বের করা
 */
function getOrdersSheet(ss) {
  var candidates = [
    CONFIG.DEFAULT_SHEET_NAME,
    "Orders",
    "orders",
    "Jhadimadi orders",
    "Jhadimadi Orders",
    "jhadimadi orders",
    "অর্ডার",
    "Sheet1",
    "Sheet 1"
  ];

  for (var i = 0; i < candidates.length; i++) {
    var s = ss.getSheetByName(candidates[i]);
    if (s) return s;
  }

  return ss.getActiveSheet() || ss.getSheets()[0];
}

/**
 * হেল্পার ফাংশন: যদি কলাম হেডার না থাকে, তবে A-J হেডার স্বয়ংক্রিয়ভাবে বসানো
 */
function ensureHeaders(sheet) {
  var lastRow = sheet.getLastRow();
  var firstCellVal = "";
  try {
    firstCellVal = sheet.getRange(1, 1).getValue().toString().trim();
  } catch (_) {}

  if (lastRow === 0 || firstCellVal === "") {
    var headers = [
      "order ID",        // Column A
      "Date & Time",     // Column B
      "Customer Name",   // Column C
      "Phone Number",    // Column D
      "Delivery Address",// Column E
      "Product ID",      // Column F
      "Product Name",    // Column G
      "Quality/Size",    // Column H
      "Total Amount",    // Column I
      "Status"           // Column J
    ];
    if (lastRow === 0) {
      sheet.appendRow(headers);
    } else {
      sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
    }
    sheet.getRange(1, 1, 1, headers.length).setFontWeight("bold").setBackground("#f3f4f6");
    sheet.setFrozenRows(1);
  }
}

/**
 * হেল্পার ফাংশন: শিটে নিরাপদভাবে রো যুক্ত করা এবং ফোন কলামকে টেক্সট ফরম্যাট দেওয়া
 */
function appendOrderRow(sheet, rowArray) {
  sheet.appendRow(rowArray);
  var lastRow = sheet.getLastRow();
  try {
    sheet.getRange(lastRow, 4).setNumberFormat("@");
  } catch (_) {}
}

/**
 * হেল্পার ফাংশন: ফোন নাম্বার ফরম্যাটিং (0 যেন কেটে না যায়)
 */
function formatPhoneNumber(phone) {
  if (!phone) return "";
  var clean = String(phone).trim();
  if (clean && !clean.startsWith("'")) {
    return "'" + clean;
  }
  return clean;
}

/**
 * স্ট্রিং স্যানিটাইজার
 */
function sanitizeString(val) {
  if (val === null || val === undefined) return "";
  return String(val).trim();
}

/**
 * স্ট্যান্ডার্ড JSON রেসপন্স তৈরি
 */
function createJsonResponse(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
```

---

## 🚀 গুগল অ্যাপস স্ক্রিপ্টে কোড আপডেট করার নিয়ম (URL পরিবর্তন ছাড়া)

> **খুব গুরুত্বপূর্ণ:** আপনার বর্তমান Web App URL যেন পরিবর্তন না হয় এবং আগের সব সেটিংস ঠিক থাকে, সেজন্য কখনোই **New Deployment** করবেন না। নিচের ধাপগুলো অনুসরণ করে **Manage Deployments** দিয়ে আপডেট করুন:

### ধাপ ১: স্ক্রিপ্ট এডিটরে কোড পেস্ট করুন
১. আপনার গুগল স্প্রেডশিটে যান (`Jhadimadi orders`)।
২. উপরের মেনু থেকে **Extensions (এক্সটেনশন)** > **Apps Script (অ্যাপস স্ক্রিপ্ট)** এ ক্লিক করুন।
৩. `Code.gs` ফাইলের বর্তমান সব কোড মুছে ফেলে উপরের সম্পূর্ণ নতুন কোডটি পেস্ট করুন।
৪. **Save (সেভ)** আইকনে ক্লিক করুন (অথবা কীবোর্ডে `Ctrl + S` / `Cmd + S` চাপুন)।

---

### ধাপ ২: Manage Deployments দিয়ে আপডেট করুন (একই URL বহাল থাকবে)
১. Apps Script এর উপরে ডানপাশে নীল রঙের **Deploy (ডিপ্লয়)** বাটনে ক্লিক করুন।
২. ড্রপডাউন মেনু থেকে **Manage deployments (ম্যানেজ ডিপ্লয়মেন্টস)** সিলেক্ট করুন।
৩. আপনার বিদ্যমান অ্যাক্টিভ ডিপ্লয়মেন্টের ডানপাশে থাকা **✏️ Edit (পেন্সিল আইকন)** এ ক্লিক করুন।
৪. **Version** ড্রপডাউনে ক্লিক করে **New version (নতুন ভার্সন)** নির্বাচন করুন।
   - *(ঐচ্ছিক)* Description বক্সে লিখতে পারেন: `Multi-item order fix v2`
৫. **Deploy (ডিপ্লয়)** বাটনে চাপ দিন।
৬. **Done** এ ক্লিক করুন।

✅ **ব্যাস, কাজ শেষ!** আপনার বিদ্যমান Web App URL অপরিবর্তিত থাকবে এবং নতুন সব ফিচার (সিঙ্গেল আইটেম + মাল্টি-আইটেম কার্ট অর্ডার) তাৎক্ষণিকভাবে কার্যকর হয়ে যাবে।

---

## 🔗 প্রকল্পের বর্তমান কার্যকর Web App URL:
```text
https://script.google.com/macros/s/AKfycbwQ4lBNjT5cIetA3AhKFPNRtgtAsCJVzssgSAbsnbnln09LGxshrpJ4tqpaWPTWk4ATdw/exec
```
ফ্রন্টএন্ড চেকআউট থেকে সরাসরি এই URL-এ প্রতিটি মাল্টি-আইটেম অর্ডারের জন্য `items` অ্যারে পাঠানো হয় যাতে গুগল শিটের `orders` ট্যাবে প্রতিটি পণ্য আলাদা আলাদা রো-তে (row) সুন্দরভাবে লিপিবদ্ধ হয়।
