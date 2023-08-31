const k = 5 + 1;  //近い駅探す件数
const CSV_URL = "https://raw.githubusercontent.com/atsushi-green/station2vec/main/scripts/distance_matrix_release.csv"

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

let distanceMatrix_csv = read_distance_matrix(CSV_URL)

let distanceMatrix = distanceMatrix_csv.slice(1)  //ヘッダーを除く

var targetStation = ""
var element = document.getElementById('targetStation');

function butotnClick() {
    // ボタンが押下された時に、近しい駅を表示する
    targetStation = nameText.value
    msg.innerText = targetStation + "に近い駅は";
    msg2.innerText = "です。";
    var index = station2index[targetStation];

    var SmallestIndexes = getSmallestIndexes(distanceMatrix[index], k)
    var nearStations = "<ol>"
    for (let i = 1; i < k; i++) {
        nearStations = nearStations + "<li>" + stations[SmallestIndexes[i]] + "</li>"
    }
    nearStations = nearStations + "</ol>"
    // msg.innerText = nearStations
    document.getElementById('a').innerHTML = nearStations;

}


let nameText = document.getElementById('targetStation');
let msg = document.getElementById('msg');

let checkButton = document.getElementById('searchButton');
checkButton.addEventListener('click', butotnClick);






// 駅名からインデックスへの連想配列を作る
var station2index = new Object();
let stations = distanceMatrix_csv[0]
for (let i = 0; i < stations.length; i++) {
    station2index[stations[i]] = i;
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
