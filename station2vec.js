const k = 5 + 1;  //近い駅探す件数（5件欲しいが、同一駅の距離が0で最小なので、+1件多く取得する）
const CSV_URL = "https://raw.githubusercontent.com/atsushi-green/station2vec/main/scripts/distance_matrix_release.csv"
var targetStation = ""  // この駅に近い駅を探す
var stationText = document.getElementById('targetStation');  // テキストボックスオブジェクト
let msg = document.getElementById('msg');  // "{targetStation}に近い駅は" の文字列


function read_distance_matrix(url) {
    // CSVファイルを取得
    let csv = new XMLHttpRequest();
    // CSVファイルへのパス
    csv.open("GET", url, false);

    // csvファイル読み込み失敗時のエラー対応
    try {
        csv.send(null);
    } catch (err) {
        console.log(err);
    }

    // 配列を定義
    let csvArray = [];

    // 改行ごとに配列化
    let lines = csv.responseText.split(/\r\n|\n/);

    // 1行ごとに処理
    for (let i = 0; i < lines.length; ++i) {
        let cells = lines[i].split(",");
        if (cells.length != 1) {
            csvArray.push(cells);
        }
    }
    return csvArray
}


// 駅間のユークリッド距離をGitHub上のcsvファイルから読み込み
let distanceMatrix_csv = read_distance_matrix(CSV_URL);
let stations = distanceMatrix_csv[0];  //先頭行(ヘッダー)が駅名
let distanceMatrix = distanceMatrix_csv.slice(1);  //ヘッダーを除くと距離行列


// 駅名からインデックスへの連想配列を作る
var station2index = new Object();
for (let i = 0; i < stations.length; i++) {
    station2index[stations[i]] = i;
}
function sanitize_string(string) {
    // 入力テキストのサニタイズ処理
    string = string.replace(/</g, "&lt;");
    string = string.replace(/>/g, "&gt;");
    string = string.replace(/"/g, "&quot;");
    string = string.replace(/'/g, "&#39;");
    string = string.replace(/&/g, "&amp;");
    return string;
}
function butotnClick() {
    // ボタンが押下された時に、近しい駅を表示する
    targetStation = stationText.value
    targetStation = sanitize_string(targetStation)
    if (stations.includes(targetStation)) {

        msg.innerText = targetStation + " に似ている駅は";
        msg2.innerText = "です。";
    } else {
        msg.innerText = targetStation + " は今回のデータには含まれていません。";
        msg2.innerText = "\n";
        document.getElementById('near_station').innerHTML = "";

        return
    }
    var index = station2index[targetStation];

    var SmallestIndexes = getSmallestIndexes(distanceMatrix[index], k)
    var nearStations = "<ol>"
    for (let i = 1; i < k; i++) {
        nearStations = nearStations + "<li>" + stations[SmallestIndexes[i]] + "</li>"
    }
    nearStations = nearStations + "</ol>"
    document.getElementById('near_station').innerHTML = nearStations;
    return

}

function getSmallestIndexes(arr, k) {
    // 上位k件の最小値を持つ要素のインデックスを返す
    const indexes = [];

    for (let i = 0; i < k; i++) {
        let minValue = Infinity;
        let minIndex = -1;

        for (let j = 0; j < arr.length; j++) {
            if (indexes.includes(j)) {
                continue; // Skip if index already selected
            }

            if (arr[j] < minValue) {
                minValue = arr[j];
                minIndex = j;
            }
        }

        indexes.push(minIndex);
    }

    return indexes;
}


// ボタン押下の取得
let checkButton = document.getElementById('searchButton');
checkButton.addEventListener('click', butotnClick);
butotnClick();  // 初期表示