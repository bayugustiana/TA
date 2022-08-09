<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Aktivitas;
use Illuminate\Support\Facades\DB;
use App\Models\Sapi;
use App\Models\Tempat;

class SapiController extends Controller
{
    public function index()
    {
        $data_get = Aktivitas::orderBy('waktu', 'DESC')->groupBy('id_sapi')->with('sapi.tempat.peternakan')->whereHas('sapi.tempat', function($q){$q->where('id_peternakan', 1);})->get();
        $data_get_array = $data_get->toArray();
        if (@$data_get) {
            $ml_data = getMLCountSapi($data_get_array);
            $ml_data = countGroupMLValues($ml_data);
            $data_kesehatan = MLGraphFormat($ml_data);
        } else {
            $data_kesehatan = null;
        }

        $data = array(
            'info_kesehatan' => $data_kesehatan,
            'info_sapi' => $data_get,
            'title' => 'Sapi'
        );
        return view('peternak.pages.sapi.index')->with($data);
    }

    public function aktivitas()
    {
        $data_get = Aktivitas::whereIn('id', function($query){$query->selectRaw('MAX(id)')->from('aktivitas')->groupBy('id_sapi');})->orderBy('waktu', 'desc')->with('sapi.tempat.peternakan')->whereHas('sapi.tempat', function($q){$q->where('id_peternakan', 1);})->get()->toArray();
        $data_get2 = (object) $data_get[0]['sapi']['tempat']['peternakan'];
        $data_get3 = json_decode(json_encode($data_get), FALSE);
        if (@$data_get) {
            $ml_data = getMLCountSapi($data_get);
            $ml_data = MergeSapiWithMLValues($ml_data,$data_get);
        } else {
            $ml_data = null;
        }
        $data = array(
            'info_aktivitas' => $ml_data,
            'info_peternakan' => $data_get2,
            'info_lokasi' => $data_get3,
            'title' => 'Aktivitas Kesehatan Sapi'
        );
        return view('peternak.pages.sapi.aktivitas')->with($data);
    }

    public function create()
    {
        $data_get = Tempat::all();
        $data = array(
            'info_tempat' => $data_get,
            'title' => 'Tambah Sapi Baru'
        );
        return view('peternak.pages.sapi.form')->with($data);
    }

    public function store(Request $request)
    {
        $request->validate([
            'id_tempat' => 'required',
            'tanggal_lahir' => 'required',
            'jenis_sapi' => 'required',
            'tipe_sapi' => 'required',
            'jenis_kelamin' => 'required'
        ]);

        $data_post = new Sapi([
            'id_tempat' => $request->get('id_tempat'),
            'tanggal_lahir' => $request->get('tanggal_lahir'),
            'jenis_sapi' => $request->get('jenis_sapi'),
            'tipe_sapi'=> $request->get('tipe_sapi'),
            'jenis_kelamin' => $request->get('jenis_kelamin')
        ]);
        $data_post->save();
        return redirect(route('sapi.index'))->with('success', 'Sapi baru berhasil ditambahkan!');
    }

    public function show($id)
    {
        $data = array(
            'title' => 'Tampil Sapi #'.$id
        );
        return view('peternak.pages.sapi.show')->with($data);
    }

    public function edit($id)
    {
        $data_get = Sapi::find($id);
        $data_get2 = Tempat::where('id_peternakan', 1)->get();
        $data = array(
            'info' => $data_get,
            'info_tempat' => $data_get2,
            'title' => 'Edit Sapi #'.$id
        );
        return view('peternak.pages.sapi.form')->with($data);
    }

    public function update(Request $request, $id)
    {
        $request->validate([
            'id_tempat' => 'required',
            'tanggal_lahir' => 'required',
            'jenis_sapi' => 'required',
            'tipe_sapi' => 'required',
            'jenis_kelamin' => 'required'
        ]);

        $info = Sapi::find($id);

        $info->id_tempat = $request->get('id_tempat');
        $info->tanggal_lahir = $request->get('tanggal_lahir');
        $info->jenis_sapi = $request->get('jenis_sapi');
        $info->tipe_sapi = $request->get('tipe_sapi');
        $info->jenis_kelamin = $request->get('jenis_kelamin');
        $info->save();

        return redirect(route('sapi.index'))->with('success', 'Sapi #'.$id.' berhasil diperbaharui!');
    }

    public function destroy($id)
    {
        $info = Sapi::find($id);
        return redirect(route('sapi.index'))->with('success', 'Sapi #'.$id.' berhasil dihapuskan!');
    }
}
