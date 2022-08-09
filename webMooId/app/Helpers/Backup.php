<?php

function filterArrayByKeys(array $input, array $column_keys)
{
    $result = array_filter($input, function ($k) use ($column_keys) {
        return in_array($k, $column_keys);
    }, ARRAY_FILTER_USE_KEY);
    return $result;
}

function getMLCountSapi($request)
{
    $request = filterArrayByKeys($request, ['suhu','detak_jantung']);
    foreach($request as $key => $csm)
    {
        $request['status'] = 0;
    }

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


// --------------------------------------------

// $data_get = Aktivitas::orderBy('waktu', 'DESC')->groupBy('id_sapi')->with('sapi.tempat')->whereHas('sapi.tempat', function($q){$q->where('id_peternakan', 1);})->get()->toArray();
        // $data_get = DB::table('tempat')->rightJoin('sapi', 'tempat.id', '=', 'sapi.id_tempat')->rightJoin('aktivitas', 'sapi.id', '=', 'aktivitas.id_sapi')->select('tempat.id', 'tempat.jenis_tempat', 'sapi.*', 'aktivitas.id as id_aktivitas', 'aktivitas.*')->where('tempat.id_peternakan', '1')->get()->toArray();
        $data_get = Tempat::with(['sapi.aktivitas' => function($query){$query->whereIn('id', function($query){$query->selectRaw('MAX(id)')->from('aktivitas')->groupBy('id_sapi');})->orderBy('waktu', 'desc');}])->where('id_peternakan', 1)->get()->toArray();
        // dd($data_get[0]['sapi'][0]['aktivitas'][0]['latitude']);
        // dd($data_get);
        $group_mldata = array();
        $i = 0;
        if (@$data_get) {
            foreach($data_get as $data_tempat => $value) {
                foreach($data_get[$data_tempat]['sapi'] as $data_sapi[$i]) {
                    if($data_sapi[$i]['aktivitas'] != null) {
                        $ml_data = getMLCountSapi($data_sapi[$i]['aktivitas']);
                        $ml_data = MergeSapiWithMLValues($ml_data,$data_sapi[$i]['aktivitas']);
                        //Script for adding this array to group array
                        $group_mldata = position_merge($ml_data, $group_mldata, 'id_sapi');
                        // $data_get[$data_tempat]['sapi'][$data_sapi]['aktivitas'][0] = $ml_data;
                        $data_get[$data_tempat]['sapi'][$i]['aktivitas'] = $group_mldata;
                    }
                    $i++;
                }
            }
        } else {
            $ml_data = null;
        }

        dd($data_get);
