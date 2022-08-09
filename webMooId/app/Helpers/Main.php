<?php

// ------------------------------------------------------------------
// REQUIRED LIBRARY FUNCTIONS
// ------------------------------------------------------------------

function replaceKeys($oldKey, $newKey, array $input){
    $return = array();
    foreach ($input as $key => $value) {
        if ($key===$oldKey)
            $key = $newKey;

        if (is_array($value))
            $value = replaceKeys( $oldKey, $newKey, $value);

        $return[$key] = $value;
    }
    return $return;
}


function filterArrayByKeys(array $input, array $column_keys)
{
    $result = array();
    $column_keys = array_flip($column_keys); // getting keys as values
    foreach ($input as $key => $val) {
        // getting only those key value pairs, which matches $column_keys
        $result[$key] = array_intersect_key($val, $column_keys);
    }
    return $result;
}

function changeKondisiValue($request)
{
    foreach($request as $key => $value)
    {
        switch($request[$key]['kondisi'])
        {
            case 0:
                $request[$key]['kondisi'] = 'Aman';
                break;
            case 1:
                $request[$key]['kondisi'] = 'Peringatan';
                break;
            case 2:
                $request[$key]['kondisi'] = 'Berbahaya';
                break;
            case 3:
                $request[$key]['kondisi'] = 'Sangat Berbahaya';
                break;
        }
    }
    return $request;
}

function position_merge($arr1,$arr2,$column)
{
    $result=array_merge($arr1,$arr2);
    array_walk($result,function(&$item,$idx){
        $item["array_index"]=$idx;// Make sure this key is not used
    });
    usort($result,function($a,$b) use ($column){
        return $a[$column]-$b[$column]?:$a["array_index"]-$b["array_index"];
    });
    array_walk($result,function(&$item){
        unset($item["array_index"]);
    });
    return $result;
}

// ------------------------------------------------------------------

function getMLCountSapi($request)
{
    $request = filterArrayByKeys($request, ['suhu','detak_jantung']);
    $request = array_map(function($row) {
        $row['status'] = 0;
        return $row;
    }, $request);
    $data = array (
        'Inputs' => array(),
        'GlobalParameters' => null,
    );
    $data["Inputs"]["input1"] = $request;

    $body = json_encode($data);

    $url = 'https://ussouthcentral.services.azureml.net/workspaces/ba33897613f340d593993544cf98c424/services/201a0b6cd8f74e88952fddb35aa25856/execute?api-version=2.0&format=swagger';
    $api_key = 'OGBx/z5j983DtOv/llkcv2sP3cvl92DFTww9odDGdQGEJ6MH81TWSJLLsO3/29R1D7rc6KCxFLpCaeH2FAjs3A==';
    $headers = array('Content-Type: application/json', 'Authorization:Bearer ' . $api_key);

    $curl = curl_init();

    curl_setopt($curl, CURLOPT_URL,$url);
    curl_setopt($curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_setopt($curl, CURLOPT_POSTFIELDS, $body);
    curl_setopt($curl, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);

    $result = curl_exec($curl);
    curl_close($curl);

    $response = json_decode($result, true);
    $mldata = $response['Results']['output1'];

    return $mldata;
}

function countGroupMLValues($request)
{
    $mldata = $request;
    $mldata = array_column($mldata, 'Scored Labels');
    $mldata = array_count_values($mldata);
    return $mldata;
}

function MergeSapiWithMLValues($request,$data_sapi)
{
    $request = replaceKeys('Scored Labels', 'kondisi', $request);
    $request = changeKondisiValue($request);
    foreach($request as $key => $csm)
    {
        $data_sapi[$key]['kondisi'] = $request[$key]['kondisi'];
    }
    return $data_sapi;
}

function MLGraphFormat($input)
{
    $input = replaceKeys(3, 'Sangat Berbahaya', $input);
    $input = replaceKeys(2, 'Berbahaya', $input);
    $input = replaceKeys(1, 'Peringatan', $input);
    $input = replaceKeys(0, 'Aman', $input);

    $finalformat = array(array());

    $i = 0;
    foreach($input as $index=>$value)
    {
        $finalformat[$i]['type'] = $index;
        $finalformat[$i]['count'] = $value;
        $i++;
    }
    return json_encode($finalformat);
}

function checkIfThreeDayPassed($input_date)
{
    $result = false;
    $timestamp = $input_date;

    $today = new DateTime(); // This object represents current date/time
    $today->setTime( 0, 0, 0 ); // reset time part, to prevent partial comparison

    $match_date = DateTime::createFromFormat( "Y-m-d H:i:s", $timestamp );
    $match_date->setTime( 0, 0, 0 ); // reset time part, to prevent partial comparison


    $diff = $today->diff( $match_date );
    $diffDays = (integer)$diff->format( "%R%a" ); // Extract days count in interval

    if($diffDays <= -2) {
        $result = true;
    }

    return $result;
}

function timeForHuman($input_date, $status_hms = NULL) {
    $time = strtotime($input_date);
    switch($status_hms) {
        case "standar":
            $view_time = date("d M Y", $time);
            break;
        default:
            $view_time = date("d M Y - H:m:s", $time);
            break;
    }
    return $view_time;
}
