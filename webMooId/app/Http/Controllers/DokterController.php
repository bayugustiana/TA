<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Dokter;

class DokterController extends Controller
{
    public function index()
    {
        $data_get = Dokter::where('id_peternakan', '1')->get();
        $data = array(
            'info_dokter' => $data_get,
            'title' => 'Dokter'
        );
        return view('peternak.pages.dokter.index')->with($data);
    }

    public function create()
    {
        $data = array(
            'title' => 'Tambah Dokter Baru'
        );
        return view('peternak.pages.dokter.form')->with($data);
    }

    public function store(Request $request)
    {
        return redirect(route('dokter.index'))->with('success', 'Dokter baru berhasil didaftarkan!');
    }

    public function show($id)
    {
        $data_get = Dokter::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'Tampil Dokter #'.$id
        );
        return view('peternak.pages.dokter.show')->with($data);
    }

    public function edit($id)
    {
        $data_get = Dokter::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'Edit Dokter #'.$id
        );
        return view('peternak.pages.dokter.form')->with($data);
    }

    public function update(Request $request, $id)
    {
        return redirect(route('dokter.index'))->with('success', 'Dokter #'.$id.' berhasil diubah!');
    }

    public function destroy(Request $request, $id)
    {
        return redirect(route('dokter.index'))->with('success', 'Dokter #'.$id.' berhasil dihapus!');
    }
}
