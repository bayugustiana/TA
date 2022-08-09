<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Peternakan;
use App\Models\Tempat;
use App\Models\Sapi;
use App\Models\Aktivitas;

class TempatController extends Controller
{
    public function index()
    {
        $data_get = Tempat::where('id_peternakan', '1')->get();
        $data_get2 = Peternakan::find(1);
        $data = array(
            'info_tempat' => $data_get,
            'info_peternakan' => $data_get2,
            'title' => 'Tempat'
        );
        return view('peternak.pages.tempat.index')->with($data);
    }

    public function create()
    {
        $data = array(
            'title' => 'Tambah Tempat Baru'
        );
        return view('peternak.pages.tempat.form')->with($data);
    }

    public function store(Request $request)
    {
        return redirect(route('tempat.index'))->with('success', 'Tempat baru berhasil ditambahkan!');
    }

    public function show($id)
    {
        $data_get = Tempat::find($id);
        $data_get2 = Sapi::where('id_tempat', $id)->get();
        $data_get_query = Aktivitas::orderBy('waktu', 'DESC')->groupBy('id_sapi')->with('sapi')->whereHas('sapi', function($q) use ($id){$q->where('id_tempat', $id);})->get();
        $data_get_array = $data_get_query->toArray();
        if (@$data_get_array) {
            $ml_data = getMLCountSapi($data_get_array);
            $ml_data = MergeSapiWithMLValues($ml_data,$data_get_array);
            $data_get3 = $ml_data;
        } else {
            $data_get3 = null;
        }
        $data = array(
            'info' => $data_get,
            'info_lokasi' => $data_get2,
            'info_sapi' => $data_get3,
            'title' => 'Tampil Tempat #'.$id
        );
        return view('peternak.pages.tempat.show')->with($data);
    }

    public function edit($id)
    {
        $data_get = Tempat::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'Edit Tempat #'.$id
        );
        return view('peternak.pages.tempat.form')->with($data);
    }

    public function update(Request $request, $id)
    {
        return redirect(route('tempat.index'))->with('success', 'Tempat #'.$id.' berhasil diubah!');
    }

    public function destroy(Request $request, $id)
    {
        return redirect(route('tempat.index'))->with('success', 'Tempat #'.$id.' berhasil dihapus!');
    }

    public function create_gateway()
    {
        $data = array(
            'title' => 'Tambah Gateway Baru'
        );
        return view('peternak.pages.tempat.form_gateway')->with($data);
    }

    public function store_gateway(Request $request)
    {
        return redirect(route('tempat.index'))->with('success', 'Gateway baru berhasil ditambahkan!');
    }

    public function edit_gateway($id)
    {
        $data = array(
            'title' => 'Edit Gateway #'.$id
        );
        return view('peternak.pages.tempat.form_gateway')->with($data);
    }

    public function update_tempat(Request $request, $id)
    {
        return redirect(route('tempat.index'))->with('success', 'Gateway #'.$id.' berhasil diubah!');
    }

    public function destroy_tempat(Request $request, $id)
    {
        return redirect(route('tempat.index'))->with('success', 'Gateway #'.$id.' berhasil dihapus!');
    }

}
